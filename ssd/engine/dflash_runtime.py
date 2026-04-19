from dataclasses import dataclass
from time import perf_counter

import torch
from transformers import AutoModel, AutoModelForCausalLM, DynamicCache

from ssd.config import Config
from ssd.engine.dflash_predictor import DFlashFeaturePredictor
from ssd.layers.sampler import Sampler
from ssd.utils.async_helpers.async_spec_helpers import get_forked_recovery_tokens_from_logits


@dataclass
class DFlashBlockOutputs:
    draft_tokens: torch.Tensor
    logits_q: torch.Tensor
    block_hidden: torch.Tensor
    branch_logits: torch.Tensor
    predicted_target_features: torch.Tensor | None
    dflash_time_s: float = 0.0
    predictor_time_s: float = 0.0


@dataclass
class DFlashCommittedState:
    exact_history: torch.Tensor
    frontier_version: int = 0


@dataclass
class DFlashCacheEntry:
    tokens: torch.Tensor
    logits_q: torch.Tensor
    branch_logits: torch.Tensor
    predicted_target_features: torch.Tensor | None
    branch_rank: int | None = None


@dataclass
class DFlashBranchCachePopulateResult:
    cache: dict[tuple[int, int, int, int], DFlashCacheEntry]
    num_branches_generated: list[int]
    dflash_time_s: float = 0.0
    predictor_time_s: float = 0.0


@dataclass
class DFlashRealizedBranchDiagnostics:
    actual_accept_supported: list[bool]
    actual_recovery_rank_given_accept: list[int | None]
    joint_branch_supported: list[bool]
    recovery_entropy_at_actual_accept: list[float | None]
    recovery_top1_margin_at_actual_accept: list[float | None]


class DFlashRuntime:
    def __init__(
        self,
        config: Config,
        device: torch.device,
        predictor_path: str | None = None,
        metrics: dict | None = None,
    ):
        self.config = config
        self.device = device
        self.dtype = getattr(config.draft_hf_config, "torch_dtype", None) or torch.bfloat16
        self.block_size = config.dflash_block_size
        self.lookahead = config.speculate_k
        self.mask_token_id = config.dflash_mask_token_id
        self.feature_dim = config.dflash_target_feature_dim
        self.metrics = metrics
        self.base_states: dict[int, DFlashCommittedState] = {}
        self.predicted_frontiers: dict[int, tuple[int, torch.Tensor]] = {}

        self.draft_model = AutoModel.from_pretrained(
            config.draft,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(device).eval()
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=self.dtype,
        ).to(device).eval()
        self.predictor = None
        if predictor_path is not None:
            self.predictor = DFlashFeaturePredictor.from_pretrained(
                predictor_path,
                device=device,
                dtype=self.dtype,
            ).eval()
        self.sampler = Sampler(config.sampler_x, config.async_fan_out).to(device)

    def reset_states(self) -> None:
        self.base_states.clear()
        self.predicted_frontiers.clear()

    def prefill_exact_context(
        self,
        seq_ids: list[int],
        prompt_target_features: list[torch.Tensor],
        frontier_version: int = 0,
    ) -> None:
        for seq_id, features in zip(seq_ids, prompt_target_features):
            self.base_states[seq_id] = DFlashCommittedState(
                exact_history=features.to(device=self.device, dtype=self.dtype, non_blocking=False).clone(),
                frontier_version=frontier_version,
            )

    def commit_exact_context(
        self,
        seq_ids: list[int],
        frontier_versions: list[int],
        exact_feature_updates: list[torch.Tensor | None],
    ) -> None:
        for seq_id, frontier_version, update in zip(seq_ids, frontier_versions, exact_feature_updates):
            state = self.base_states.get(seq_id)
            if state is None:
                raise RuntimeError(f"Missing committed DFlash state for seq_id={seq_id}")
            if update is None or update.numel() == 0:
                if state.frontier_version != frontier_version:
                    raise RuntimeError(
                        f"DFlash frontier mismatch for seq_id={seq_id}: "
                        f"state={state.frontier_version}, request={frontier_version}"
                    )
                continue

            if frontier_version != state.frontier_version + 1:
                raise RuntimeError(
                    f"DFlash exact commit out of order for seq_id={seq_id}: "
                    f"state={state.frontier_version}, request={frontier_version}"
                )
            update = update.to(device=self.device, dtype=self.dtype, non_blocking=False)
            state.exact_history = torch.cat([state.exact_history, update], dim=0).contiguous()
            state.frontier_version = frontier_version

    def get_exact_history(self, seq_id: int) -> torch.Tensor:
        state = self.base_states.get(seq_id)
        if state is None:
            raise RuntimeError(f"Missing committed DFlash state for seq_id={seq_id}")
        return state.exact_history

    def get_generation_history(
        self,
        seq_id: int,
        frontier_version: int,
        prefer_predicted: bool = False,
    ) -> tuple[torch.Tensor, bool]:
        if prefer_predicted:
            predicted = self.predicted_frontiers.get(seq_id)
            if predicted is not None and predicted[0] == frontier_version:
                self.predicted_frontiers.pop(seq_id, None)
                return predicted[1].clone(), True
        return self.get_exact_history(seq_id).clone(), False

    def _build_predicted_child_history(
        self,
        seq_id: int,
        entry: DFlashCacheEntry,
        accept_idx: int,
    ) -> torch.Tensor:
        if entry.predicted_target_features is None:
            raise RuntimeError("DFlash predicted branch construction requires predicted_target_features")
        if accept_idx < 0:
            raise ValueError(f"accept_idx must be >= 0, got {accept_idx}")
        base_history = self.get_exact_history(seq_id)
        return torch.cat(
            [base_history, entry.predicted_target_features[:accept_idx + 1]],
            dim=0,
        ).contiguous()

    def generate_block(
        self,
        seq_ids: list[int],
        recovery_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        return_predicted_features: bool = True,
    ) -> DFlashBlockOutputs:
        histories = [self.get_exact_history(seq_id).clone() for seq_id in seq_ids]
        return self.run_block_batch(
            histories,
            recovery_tokens,
            temperatures,
            return_predicted_features=return_predicted_features,
        )

    def populate_branch_cache(
        self,
        seq_ids: list[int],
        frontier_versions: list[int],
        recovery_tokens: list[int],
        temperatures: torch.Tensor,
        cache_hits: torch.Tensor,
        entries: list[DFlashCacheEntry],
    ) -> DFlashBranchCachePopulateResult:
        if not entries:
            return DFlashBranchCachePopulateResult(cache={}, num_branches_generated=[])

        branch_logits = torch.stack([entry.branch_logits for entry in entries], dim=0)
        returned_tokens = torch.stack(
            [
                torch.cat(
                    [
                        torch.tensor([recovery_tokens[i]], dtype=torch.int64, device=self.device),
                        entry.tokens,
                    ],
                    dim=0,
                )
                for i, entry in enumerate(entries)
            ],
            dim=0,
        )
        forked_recovery = get_forked_recovery_tokens_from_logits(
            self.config,
            branch_logits,
            cache_hits,
            returned_tokens,
            tokenizer=None,
        )

        jobs_by_len: dict[int, list[tuple[tuple[int, int, int, int], torch.Tensor, int, float, int, int]]] = {}
        for row_idx, entry in enumerate(entries):
            if entry.predicted_target_features is None:
                raise RuntimeError("DFlash SSD branch cache population requires predicted_target_features")
            counts = self.config.fan_out_list if bool(cache_hits[row_idx].item()) else self.config.fan_out_list_miss
            offset = 0
            base_history = self.get_exact_history(seq_ids[row_idx])
            next_frontier_version = frontier_versions[row_idx] + 1
            for accept_idx, count in enumerate(counts):
                child_history = torch.cat(
                    [base_history, entry.predicted_target_features[:accept_idx + 1]],
                    dim=0,
                ).contiguous()
                child_len = int(child_history.shape[0])
                for _ in range(count):
                    recovery_token = int(forked_recovery[row_idx, offset].item())
                    branch_rank = offset
                    offset += 1
                    cache_key = (seq_ids[row_idx], next_frontier_version, accept_idx, recovery_token)
                    jobs_by_len.setdefault(child_len, []).append(
                        (
                            cache_key,
                            child_history,
                            recovery_token,
                            float(temperatures[row_idx].item()),
                            row_idx,
                            branch_rank,
                        )
                    )

        branch_cache: dict[tuple[int, int, int, int], DFlashCacheEntry] = {}
        num_branches_generated = [0] * len(entries)
        total_dflash_s = 0.0
        total_predictor_s = 0.0
        for _child_len, jobs in jobs_by_len.items():
            histories = [history for _, history, _, _, _, _ in jobs]
            rec_tokens = torch.tensor(
                [recovery for _, _, recovery, _, _, _ in jobs],
                dtype=torch.int64,
                device=self.device,
            )
            temps = torch.tensor(
                [temp for _, _, _, temp, _, _ in jobs],
                dtype=torch.float32,
                device=self.device,
            )
            outputs = self.run_block_batch(
                histories,
                rec_tokens,
                temps,
                return_predicted_features=True,
            )
            total_dflash_s += outputs.dflash_time_s
            total_predictor_s += outputs.predictor_time_s
            for row_idx, (cache_key, _history, _recovery, _temp, source_row, branch_rank) in enumerate(jobs):
                num_branches_generated[source_row] += 1
                branch_cache[cache_key] = DFlashCacheEntry(
                    tokens=outputs.draft_tokens[row_idx].clone(),
                    logits_q=outputs.logits_q[row_idx].clone(),
                    branch_logits=outputs.branch_logits[row_idx].clone(),
                    predicted_target_features=(
                        outputs.predicted_target_features[row_idx].clone()
                        if outputs.predicted_target_features is not None else None
                    ),
                    branch_rank=branch_rank,
                )
        return DFlashBranchCachePopulateResult(
            cache=branch_cache,
            num_branches_generated=num_branches_generated,
            dflash_time_s=total_dflash_s,
            predictor_time_s=total_predictor_s,
        )

    def populate_oracle_branch_cache(
        self,
        seq_ids: list[int],
        frontier_versions: list[int],
        accepted_len_idxs: list[int],
        recovery_tokens: list[int],
        temperatures: torch.Tensor,
        entries: list[DFlashCacheEntry],
    ) -> DFlashBranchCachePopulateResult:
        if not seq_ids:
            return DFlashBranchCachePopulateResult(cache={}, num_branches_generated=[])

        jobs_by_len: dict[int, list[tuple[tuple[int, int, int, int], torch.Tensor, int, float, int]]] = {}
        for row_idx, (seq_id, frontier_version, accept_idx, recovery_token, entry) in enumerate(
            zip(seq_ids, frontier_versions, accepted_len_idxs, recovery_tokens, entries)
        ):
            child_history = self._build_predicted_child_history(seq_id, entry, accept_idx)
            child_len = int(child_history.shape[0])
            cache_key = (seq_id, frontier_version + 1, accept_idx, recovery_token)
            jobs_by_len.setdefault(child_len, []).append(
                (cache_key, child_history, recovery_token, float(temperatures[row_idx].item()), row_idx)
            )

        branch_cache: dict[tuple[int, int, int, int], DFlashCacheEntry] = {}
        num_branches_generated = [0] * len(seq_ids)
        total_dflash_s = 0.0
        total_predictor_s = 0.0
        for _child_len, jobs in jobs_by_len.items():
            histories = [history for _, history, _, _, _ in jobs]
            rec_tokens = torch.tensor(
                [recovery for _, _, recovery, _, _ in jobs],
                dtype=torch.int64,
                device=self.device,
            )
            temps = torch.tensor([temp for _, _, _, temp, _ in jobs], dtype=torch.float32, device=self.device)
            outputs = self.run_block_batch(
                histories,
                rec_tokens,
                temps,
                return_predicted_features=True,
            )
            total_dflash_s += outputs.dflash_time_s
            total_predictor_s += outputs.predictor_time_s
            for row_idx, (cache_key, _history, _recovery, _temp, source_row) in enumerate(jobs):
                num_branches_generated[source_row] += 1
                branch_cache[cache_key] = DFlashCacheEntry(
                    tokens=outputs.draft_tokens[row_idx].clone(),
                    logits_q=outputs.logits_q[row_idx].clone(),
                    branch_logits=outputs.branch_logits[row_idx].clone(),
                    predicted_target_features=(
                        outputs.predicted_target_features[row_idx].clone()
                        if outputs.predicted_target_features is not None else None
                    ),
                    branch_rank=0,
                )
        return DFlashBranchCachePopulateResult(
            cache=branch_cache,
            num_branches_generated=num_branches_generated,
            dflash_time_s=total_dflash_s,
            predictor_time_s=total_predictor_s,
        )

    def populate_exact_oracle_branch_cache(
        self,
        seq_ids: list[int],
        frontier_versions: list[int],
        accepted_len_idxs: list[int],
        recovery_tokens: list[int],
        temperatures: torch.Tensor,
        exact_feature_updates: list[torch.Tensor | None],
    ) -> DFlashBranchCachePopulateResult:
        if not seq_ids:
            return DFlashBranchCachePopulateResult(cache={}, num_branches_generated=[])

        jobs_by_len: dict[int, list[tuple[tuple[int, int, int, int], torch.Tensor, int, float, int]]] = {}
        for row_idx, (seq_id, frontier_version, accept_idx, recovery_token, update) in enumerate(
            zip(seq_ids, frontier_versions, accepted_len_idxs, recovery_tokens, exact_feature_updates)
        ):
            base_history = self.get_exact_history(seq_id)
            if update is None or update.numel() == 0:
                child_history = base_history.clone()
            else:
                update = update.to(device=self.device, dtype=self.dtype, non_blocking=False)
                child_history = torch.cat([base_history, update], dim=0).contiguous()
            child_len = int(child_history.shape[0])
            cache_key = (seq_id, frontier_version + 1, accept_idx, recovery_token)
            jobs_by_len.setdefault(child_len, []).append(
                (cache_key, child_history, recovery_token, float(temperatures[row_idx].item()), row_idx)
            )

        branch_cache: dict[tuple[int, int, int, int], DFlashCacheEntry] = {}
        num_branches_generated = [0] * len(seq_ids)
        total_dflash_s = 0.0
        for _child_len, jobs in jobs_by_len.items():
            histories = [history for _, history, _, _, _ in jobs]
            rec_tokens = torch.tensor(
                [recovery for _, _, recovery, _, _ in jobs],
                dtype=torch.int64,
                device=self.device,
            )
            temps = torch.tensor([temp for _, _, _, temp, _ in jobs], dtype=torch.float32, device=self.device)
            outputs = self.run_block_batch(
                histories,
                rec_tokens,
                temps,
                return_predicted_features=False,
            )
            total_dflash_s += outputs.dflash_time_s
            for row_idx, (cache_key, _history, _recovery, _temp, source_row) in enumerate(jobs):
                num_branches_generated[source_row] += 1
                branch_cache[cache_key] = DFlashCacheEntry(
                    tokens=outputs.draft_tokens[row_idx].clone(),
                    logits_q=outputs.logits_q[row_idx].clone(),
                    branch_logits=outputs.branch_logits[row_idx].clone(),
                    predicted_target_features=None,
                    branch_rank=0,
                )
        return DFlashBranchCachePopulateResult(
            cache=branch_cache,
            num_branches_generated=num_branches_generated,
            dflash_time_s=total_dflash_s,
            predictor_time_s=0.0,
        )

    def store_oracle_predicted_frontiers(
        self,
        seq_ids: list[int],
        frontier_versions: list[int],
        accepted_len_idxs: list[int],
        entries: list[DFlashCacheEntry],
    ) -> None:
        for seq_id, frontier_version, accept_idx, entry in zip(
            seq_ids,
            frontier_versions,
            accepted_len_idxs,
            entries,
        ):
            self.predicted_frontiers[seq_id] = (
                frontier_version + 1,
                self._build_predicted_child_history(seq_id, entry, accept_idx),
            )

    def analyze_realized_branches(
        self,
        recovery_tokens: list[int],
        cache_hits: torch.Tensor,
        accepted_len_idxs: list[int],
        realized_recovery_tokens: list[int],
        entries: list[DFlashCacheEntry],
    ) -> DFlashRealizedBranchDiagnostics:
        if not entries:
            return DFlashRealizedBranchDiagnostics([], [], [], [], [])

        branch_logits = torch.stack([entry.branch_logits for entry in entries], dim=0)
        returned_tokens = torch.stack(
            [
                torch.cat(
                    [
                        torch.tensor([recovery_tokens[i]], dtype=torch.int64, device=self.device),
                        entry.tokens,
                    ],
                    dim=0,
                )
                for i, entry in enumerate(entries)
            ],
            dim=0,
        )
        forked_recovery = get_forked_recovery_tokens_from_logits(
            self.config,
            branch_logits,
            cache_hits,
            returned_tokens,
            tokenizer=None,
        )

        masked_logits = branch_logits.float().clone()
        masked_logits[:, :-1, :] = masked_logits[:, :-1, :].scatter(
            dim=2,
            index=returned_tokens[:, 1:].unsqueeze(2),
            value=float("-inf"),
        )
        masked_log_probs = torch.log_softmax(masked_logits, dim=-1)
        masked_probs = masked_log_probs.exp()
        entropy_terms = torch.where(
            masked_probs > 0,
            masked_probs * masked_log_probs,
            torch.zeros_like(masked_probs),
        )
        entropy = -entropy_terms.sum(dim=-1)
        top2_probs = torch.topk(masked_probs, k=2, dim=-1).values

        actual_accept_supported: list[bool] = []
        actual_recovery_rank_given_accept: list[int | None] = []
        joint_branch_supported: list[bool] = []
        recovery_entropy_at_actual_accept: list[float | None] = []
        recovery_top1_margin_at_actual_accept: list[float | None] = []

        for row_idx, accept_idx in enumerate(accepted_len_idxs):
            if not (0 <= accept_idx <= self.lookahead):
                raise RuntimeError(
                    f"accepted_len_idx must be in [0, {self.lookahead}], got {accept_idx}"
                )

            counts = self.config.fan_out_list if bool(cache_hits[row_idx].item()) else self.config.fan_out_list_miss
            support = counts[accept_idx] > 0
            actual_accept_supported.append(bool(support))
            recovery_entropy_at_actual_accept.append(float(entropy[row_idx, accept_idx].item()))
            recovery_top1_margin_at_actual_accept.append(
                float((top2_probs[row_idx, accept_idx, 0] - top2_probs[row_idx, accept_idx, 1]).item())
            )

            if not support:
                actual_recovery_rank_given_accept.append(None)
                joint_branch_supported.append(False)
                continue

            offset = sum(counts[:accept_idx])
            count = counts[accept_idx]
            candidate_tokens = forked_recovery[row_idx, offset:offset + count].tolist()
            realized_token = realized_recovery_tokens[row_idx]

            rank = None
            for candidate_rank, token_id in enumerate(candidate_tokens):
                if token_id == realized_token:
                    rank = candidate_rank
                    break

            actual_recovery_rank_given_accept.append(rank)
            joint_branch_supported.append(rank is not None)

        return DFlashRealizedBranchDiagnostics(
            actual_accept_supported=actual_accept_supported,
            actual_recovery_rank_given_accept=actual_recovery_rank_given_accept,
            joint_branch_supported=joint_branch_supported,
            recovery_entropy_at_actual_accept=recovery_entropy_at_actual_accept,
            recovery_top1_margin_at_actual_accept=recovery_top1_margin_at_actual_accept,
        )

    @torch.inference_mode()
    def run_block_batch(
        self,
        feature_histories: list[torch.Tensor],
        recovery_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        return_predicted_features: bool = True,
    ) -> DFlashBlockOutputs:
        batch_size = len(feature_histories)
        if batch_size == 0:
            raise ValueError("run_block_batch requires at least one feature history")
        if recovery_tokens.shape != (batch_size,):
            raise ValueError(
                f"Expected recovery_tokens with shape ({batch_size},), got {tuple(recovery_tokens.shape)}"
            )
        if temperatures.shape != (batch_size,):
            raise ValueError(
                f"Expected temperatures with shape ({batch_size},), got {tuple(temperatures.shape)}"
            )

        hist_lens = [int(history.shape[0]) for history in feature_histories]
        if any(history.shape[-1] != self.feature_dim for history in feature_histories):
            raise ValueError("All feature histories must match dflash_target_feature_dim")

        vocab_size = self.target_model.lm_head.weight.shape[0]
        hidden_size = getattr(self.draft_model.config, "hidden_size")
        draft_tokens = torch.empty((batch_size, self.lookahead), dtype=torch.int64, device=self.device)
        logits_q = torch.empty((batch_size, self.lookahead, vocab_size), dtype=self.dtype, device=self.device)
        block_hidden = torch.empty((batch_size, self.block_size, hidden_size), dtype=self.dtype, device=self.device)
        branch_logits = torch.empty((batch_size, self.block_size, vocab_size), dtype=self.dtype, device=self.device)
        predicted_target_features = None
        if return_predicted_features and self.predictor is not None:
            predicted_target_features = torch.empty(
                (batch_size, self.block_size, self.feature_dim),
                dtype=self.dtype,
                device=self.device,
            )

        buckets: dict[int, list[int]] = {}
        for idx, hist_len in enumerate(hist_lens):
            buckets.setdefault(hist_len, []).append(idx)

        t0 = perf_counter()
        predictor_total_s = 0.0
        for hist_len, indices in buckets.items():
            histories = torch.stack(
                [feature_histories[idx].to(device=self.device, dtype=self.dtype, non_blocking=False) for idx in indices],
                dim=0,
            )
            rec_tokens = recovery_tokens[indices]
            temps = temperatures[indices]

            block_output_ids = torch.full(
                (len(indices), self.block_size),
                self.mask_token_id,
                dtype=torch.long,
                device=self.device,
            )
            block_output_ids[:, 0] = rec_tokens
            noise_embedding = self.target_model.model.embed_tokens(block_output_ids)
            position_ids = torch.arange(0, hist_len + self.block_size, device=self.device).unsqueeze(0).expand(len(indices), -1)
            draft_hidden = self.draft_model(
                target_hidden=histories,
                noise_embedding=noise_embedding,
                position_ids=position_ids,
                past_key_values=DynamicCache(),
                use_cache=True,
                is_causal=False,
            )
            bucket_block_hidden = draft_hidden[:, -self.block_size:, :].contiguous()
            bucket_branch_logits = self.target_model.lm_head(bucket_block_hidden).contiguous()
            bucket_logits_q = bucket_branch_logits[:, 1:, :].contiguous()
            flat_temps = temps.repeat_interleave(self.lookahead)
            bucket_tokens = self.sampler(
                bucket_logits_q.view(-1, vocab_size),
                flat_temps,
            ).view(len(indices), self.lookahead)
            bucket_predicted = None
            if predicted_target_features is not None and self.predictor is not None:
                pt0 = perf_counter()
                bucket_predicted = self.predictor(bucket_block_hidden).to(self.dtype)
                pred_elapsed = perf_counter() - pt0
                if self.metrics is not None:
                    self.metrics["dflash_predictor_times"].append(pred_elapsed)
                predictor_total_s += pred_elapsed

            for bucket_row, out_idx in enumerate(indices):
                draft_tokens[out_idx] = bucket_tokens[bucket_row]
                logits_q[out_idx] = bucket_logits_q[bucket_row]
                block_hidden[out_idx] = bucket_block_hidden[bucket_row]
                branch_logits[out_idx] = bucket_branch_logits[bucket_row]
                if predicted_target_features is not None and bucket_predicted is not None:
                    predicted_target_features[out_idx] = bucket_predicted[bucket_row]

        total_dflash_s = perf_counter() - t0
        if self.metrics is not None:
            self.metrics["dflash_draft_step_times"].append(total_dflash_s)
        return DFlashBlockOutputs(
            draft_tokens=draft_tokens,
            logits_q=logits_q,
            block_hidden=block_hidden,
            branch_logits=branch_logits,
            predicted_target_features=predicted_target_features,
            dflash_time_s=total_dflash_s,
            predictor_time_s=predictor_total_s,
        )
