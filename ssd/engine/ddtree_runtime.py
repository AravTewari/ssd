from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from ssd.config import Config
from ssd.engine.dflash_runtime import DFlashBlockOutputs, DFlashRuntime
from ssd.engine.helpers.ddtree import (
    DDTreeEntry,
    DDTreeFrontierCandidate,
    build_best_first_tree,
    build_frontier_candidates,
    pack_tree_nodes,
)


@dataclass
class DDTreeBatchOutputs:
    entries: list[DDTreeEntry]
    dflash_time_s: float = 0.0
    predictor_time_s: float = 0.0
    tree_build_time_s: float = 0.0


@dataclass
class DDTreeCandidateJob:
    seq_id: int
    frontier_version: int
    frontier_token_ids: tuple[int, ...]
    accepted_len_idx: int
    recovery_token: int
    score: float
    history: torch.Tensor
    source_rank: int


class DDTreeRuntime:
    def __init__(
        self,
        config: Config,
        device: torch.device,
        predictor_path: str | None = None,
        metrics: dict | None = None,
    ):
        self.config = config
        self.device = device
        self.tree_budget = config.ddtree_tree_budget
        self.frontier_count = config.ddtree_frontier_count
        self.metrics = metrics
        self.dflash = DFlashRuntime(
            config=config,
            device=device,
            predictor_path=predictor_path,
            metrics=metrics,
        )

    def reset_states(self) -> None:
        self.dflash.reset_states()

    def prefill_exact_context(
        self,
        seq_ids: list[int],
        prompt_target_features: list[torch.Tensor],
        frontier_version: int = 0,
    ) -> None:
        self.dflash.prefill_exact_context(seq_ids, prompt_target_features, frontier_version=frontier_version)

    def commit_exact_context(
        self,
        seq_ids: list[int],
        frontier_versions: list[int],
        exact_feature_updates: list[torch.Tensor | None],
    ) -> None:
        self.dflash.commit_exact_context(seq_ids, frontier_versions, exact_feature_updates)

    def get_exact_history(self, seq_id: int) -> torch.Tensor:
        return self.dflash.get_exact_history(seq_id)

    def get_generation_history(
        self,
        seq_id: int,
        frontier_version: int,
        prefer_predicted: bool = False,
    ) -> tuple[torch.Tensor, bool]:
        return self.dflash.get_generation_history(
            seq_id=seq_id,
            frontier_version=frontier_version,
            prefer_predicted=prefer_predicted,
        )

    def store_oracle_predicted_frontiers(
        self,
        seq_ids: list[int],
        frontier_versions: list[int],
        accepted_len_idxs: list[int],
        entries: list[DDTreeEntry],
    ) -> None:
        dflash_entries = []
        for entry in entries:
            from ssd.engine.dflash_runtime import DFlashCacheEntry

            dflash_entries.append(
                DFlashCacheEntry(
                    tokens=entry.draft_tokens.clone(),
                    logits_q=entry.logits_q.clone(),
                    branch_logits=entry.branch_logits.clone(),
                    predicted_target_features=(
                        entry.predicted_target_features.clone()
                        if entry.predicted_target_features is not None else None
                    ),
                )
            )
        self.dflash.store_oracle_predicted_frontiers(
            seq_ids=seq_ids,
            frontier_versions=frontier_versions,
            accepted_len_idxs=accepted_len_idxs,
            entries=dflash_entries,
        )

    def _entry_from_row(
        self,
        recovery_token: int,
        outputs: DFlashBlockOutputs,
        row_idx: int,
    ) -> DDTreeEntry:
        logits_q = outputs.logits_q[row_idx]
        nodes = build_best_first_tree(logits_q, tree_budget=self.tree_budget)
        frontier_candidates = build_frontier_candidates(
            recovery_token=recovery_token,
            branch_logits=outputs.branch_logits[row_idx],
            nodes=nodes,
            frontier_count=self.frontier_count,
        )
        node_token_ids, node_depths, parents = pack_tree_nodes(
            nodes,
            tree_budget=self.tree_budget,
            device=self.device,
        )
        return DDTreeEntry(
            recovery_token=recovery_token,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            parents=parents,
            num_nodes=len(nodes),
            draft_tokens=outputs.draft_tokens[row_idx].clone(),
            logits_q=outputs.logits_q[row_idx].clone(),
            branch_logits=outputs.branch_logits[row_idx].clone(),
            predicted_target_features=(
                outputs.predicted_target_features[row_idx].clone()
                if outputs.predicted_target_features is not None else None
            ),
            frontier_candidates=frontier_candidates,
        )

    def _wrap_block_outputs(
        self,
        recovery_tokens: torch.Tensor,
        outputs: DFlashBlockOutputs,
    ) -> DDTreeBatchOutputs:
        t0 = perf_counter()
        entries = [
            self._entry_from_row(int(recovery_tokens[row_idx].item()), outputs, row_idx)
            for row_idx in range(recovery_tokens.shape[0])
        ]
        return DDTreeBatchOutputs(
            entries=entries,
            dflash_time_s=outputs.dflash_time_s,
            predictor_time_s=outputs.predictor_time_s,
            tree_build_time_s=perf_counter() - t0,
        )

    def build_exact_tree_batch(
        self,
        seq_ids: list[int],
        recovery_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        return_predicted_features: bool,
    ) -> DDTreeBatchOutputs:
        outputs = self.dflash.generate_block(
            seq_ids=seq_ids,
            recovery_tokens=recovery_tokens,
            temperatures=temperatures,
            return_predicted_features=return_predicted_features,
        )
        return self._wrap_block_outputs(recovery_tokens, outputs)

    def build_tree_batch_from_histories(
        self,
        histories: list[torch.Tensor],
        recovery_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        return_predicted_features: bool,
    ) -> DDTreeBatchOutputs:
        outputs = self.dflash.run_block_batch(
            feature_histories=histories,
            recovery_tokens=recovery_tokens,
            temperatures=temperatures,
            return_predicted_features=return_predicted_features,
        )
        return self._wrap_block_outputs(recovery_tokens, outputs)

    def build_surrogate_candidate_jobs(
        self,
        seq_id: int,
        frontier_version: int,
        entry: DDTreeEntry,
    ) -> list[DDTreeCandidateJob]:
        if entry.frontier_candidates is None:
            return []
        if entry.predicted_target_features is None:
            raise RuntimeError("DDTree surrogate frontier generation requires predicted target features")
        base_history = self.get_exact_history(seq_id)
        jobs: list[DDTreeCandidateJob] = []
        for rank, frontier in enumerate(entry.frontier_candidates):
            history = torch.cat(
                [base_history, entry.predicted_target_features[:frontier.accepted_len_idx + 1]],
                dim=0,
            ).contiguous()
            jobs.append(
                DDTreeCandidateJob(
                    seq_id=seq_id,
                    frontier_version=frontier_version + 1,
                    frontier_token_ids=frontier.frontier_token_ids,
                    accepted_len_idx=frontier.accepted_len_idx,
                    recovery_token=frontier.recovery_token,
                    score=frontier.score,
                    history=history,
                    source_rank=rank,
                )
            )
        return jobs

    def build_predicted_frontier_history(
        self,
        seq_id: int,
        entry: DDTreeEntry,
        accepted_len_idx: int,
    ) -> torch.Tensor:
        if entry.predicted_target_features is None:
            raise RuntimeError("DDTree predicted frontier history requires predicted target features")
        base_history = self.get_exact_history(seq_id)
        return torch.cat(
            [base_history, entry.predicted_target_features[:accepted_len_idx + 1]],
            dim=0,
        ).contiguous()

    def build_exact_frontier_history(
        self,
        seq_id: int,
        exact_feature_update: torch.Tensor | None,
    ) -> torch.Tensor:
        base_history = self.get_exact_history(seq_id)
        if exact_feature_update is None or exact_feature_update.numel() == 0:
            return base_history.clone()
        exact_feature_update = exact_feature_update.to(device=self.device, dtype=base_history.dtype, non_blocking=False)
        return torch.cat([base_history, exact_feature_update], dim=0).contiguous()
