import os
from dataclasses import dataclass
from time import perf_counter

import torch
import torch.distributed as dist

from ssd.config import Config
from ssd.engine.ddtree_runtime import DDTreeBatchOutputs, DDTreeRuntime
from ssd.engine.helpers.ddtree import DDTreeEntry


class DDTreeSSDCommand:
    SPECULATE = 0
    PREFILL = 1
    EXIT = 2
    POST_VERIFY = 3


@dataclass
class ActiveDDTree:
    frontier_version: int
    recovery_token: int
    entry: DDTreeEntry


@dataclass
class PendingDDTreeFeedback:
    frontier_version: int
    temperature: float
    entry: DDTreeEntry


@dataclass
class PendingCandidateEntry:
    entry: DDTreeEntry
    rank: int


class DDTreeSSDRunner:
    def __init__(self, config: Config, rank: int):
        self.config = config
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.dtype = getattr(config.draft_hf_config, "torch_dtype", None) or torch.bfloat16
        self.tree_budget = config.ddtree_tree_budget
        self.feature_dim = config.dflash_target_feature_dim

        self.active_trees: dict[int, ActiveDDTree] = {}
        self.pending_candidates: dict[int, tuple[int, dict[tuple[int, ...], PendingCandidateEntry]]] = {}
        self.pending_feedback: dict[int, PendingDDTreeFeedback] = {}

        torch.cuda.set_device(rank)
        default_port = int(os.environ.get("SSD_DIST_PORT", "1223"))
        dist.init_process_group(
            "nccl",
            f"tcp://localhost:{default_port}",
            world_size=config.num_gpus,
            rank=rank,
            device_id=self.device,
        )
        dist.new_group(ranks=[0])
        self.async_pg = dist.new_group(ranks=[0, rank])

        self.runtime = DDTreeRuntime(
            config=config,
            device=self.device,
            predictor_path=(config.dflash_predictor if config.ddtree_context_mode == "predicted" else None),
            metrics=None,
        )
        self.loop()

    def _recv_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        dist.recv(tensor, src=0, group=self.async_pg)
        return tensor

    def _send_tensor(self, tensor: torch.Tensor) -> None:
        dist.send(tensor.contiguous(), dst=0, group=self.async_pg)

    def _recv_int64(self, shape: tuple[int, ...]) -> torch.Tensor:
        return self._recv_tensor(shape, torch.int64)

    def _recv_feature_list(self, batch_size: int) -> list[torch.Tensor | None]:
        lengths = self._recv_int64((batch_size,))
        total = int(lengths.sum().item())
        flat = None
        if total > 0:
            flat = torch.empty((total, self.feature_dim), dtype=self.dtype, device=self.device)
            dist.recv(flat, src=0, group=self.async_pg)
        feature_list = []
        offset = 0
        for length in lengths.tolist():
            if length <= 0:
                feature_list.append(None)
            else:
                feature_list.append(flat[offset:offset + length].clone())
                offset += length
        return feature_list

    def _recv_suffixes(self, batch_size: int) -> list[list[int]]:
        lengths = self._recv_int64((batch_size,))
        total = int(lengths.sum().item())
        flat = self._recv_int64((total,)) if total > 0 else torch.empty((0,), dtype=torch.int64, device=self.device)
        suffixes = []
        offset = 0
        for length in lengths.tolist():
            suffixes.append(flat[offset:offset + length].tolist())
            offset += length
        return suffixes

    def _handle_prefill(self) -> None:
        batch_size = int(self._recv_int64((1,)).item())
        seq_ids = self._recv_int64((batch_size,)).tolist()
        prompt_target_features = self._recv_feature_list(batch_size)
        self.runtime.prefill_exact_context(seq_ids, prompt_target_features, frontier_version=0)
        for seq_id in seq_ids:
            self.active_trees.pop(seq_id, None)
            self.pending_candidates.pop(seq_id, None)
            self.pending_feedback.pop(seq_id, None)

    def _pack_entries(
        self,
        entries: list[DDTreeEntry],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = torch.tensor([entry.num_nodes for entry in entries], dtype=torch.int64, device=self.device)
        node_token_ids = torch.stack([entry.node_token_ids for entry in entries], dim=0)
        node_depths = torch.stack([entry.node_depths for entry in entries], dim=0)
        parents = torch.stack([entry.parents for entry in entries], dim=0)
        return num_nodes, node_token_ids, node_depths, parents

    def _send_speculate_diag(
        self,
        *,
        cache_lookup_s: float,
        service_dflash_s: float,
        service_predictor_s: float,
        service_tree_build_s: float,
        background_dflash_s: float,
        background_predictor_s: float,
        background_tree_build_s: float,
        worker_total_s: float,
        cache_hits: torch.Tensor,
        entries: list[DDTreeEntry],
    ) -> None:
        batch_diag = torch.tensor(
            [
                cache_lookup_s,
                service_dflash_s,
                service_predictor_s,
                service_tree_build_s,
                background_dflash_s,
                background_predictor_s,
                background_tree_build_s,
                worker_total_s,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        row_diag = torch.tensor(
            [
                [
                    0 if int(cache_hits[row_idx].item()) == 1 else 1,
                    0 if entries[row_idx].frontier_candidates is None else len(entries[row_idx].frontier_candidates),
                ]
                for row_idx in range(len(entries))
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._send_tensor(batch_diag)
        self._send_tensor(row_diag)

    def _build_current_entries(
        self,
        miss_indices: list[int],
        seq_ids: list[int],
        frontier_versions: list[int],
        recovery_tokens: list[int],
        temperatures: torch.Tensor,
    ) -> tuple[dict[int, DDTreeEntry], DDTreeBatchOutputs | None]:
        if not miss_indices:
            return {}, None

        histories = []
        miss_recovery_tokens = []
        miss_temps = []
        prefer_predicted = (
            self.config.ddtree_context_mode == "predicted"
            and self.config.ddtree_cache == "off"
            and self.config.ddtree_frontier_mode == "oracle"
        )
        for req_idx in miss_indices:
            if prefer_predicted:
                history, _used_predicted = self.runtime.get_generation_history(
                    seq_ids[req_idx],
                    frontier_versions[req_idx],
                    prefer_predicted=True,
                )
            else:
                history = self.runtime.get_exact_history(seq_ids[req_idx]).clone()
            histories.append(history)
            miss_recovery_tokens.append(recovery_tokens[req_idx])
            miss_temps.append(float(temperatures[req_idx].item()))

        batch = self.runtime.build_tree_batch_from_histories(
            histories=histories,
            recovery_tokens=torch.tensor(miss_recovery_tokens, dtype=torch.int64, device=self.device),
            temperatures=torch.tensor(miss_temps, dtype=torch.float32, device=self.device),
            return_predicted_features=(self.config.ddtree_context_mode == "predicted"),
        )
        return (
            {
                req_idx: batch.entries[row_idx]
                for row_idx, req_idx in enumerate(miss_indices)
            },
            batch,
        )

    def _populate_surrogate_candidates(
        self,
        seq_ids: list[int],
        frontier_versions: list[int],
        temperatures: torch.Tensor,
        entries: list[DDTreeEntry],
    ) -> tuple[float, float, float]:
        if not (
            self.config.ddtree_context_mode == "predicted"
            and self.config.ddtree_cache == "on"
            and self.config.ddtree_frontier_mode == "surrogate"
        ):
            return 0.0, 0.0, 0.0

        jobs = []
        for row_idx, (seq_id, frontier_version, entry) in enumerate(zip(seq_ids, frontier_versions, entries)):
            seq_jobs = self.runtime.build_surrogate_candidate_jobs(seq_id, frontier_version, entry)
            for job in seq_jobs:
                jobs.append((row_idx, job))

        if not jobs:
            for seq_id in seq_ids:
                self.pending_candidates.pop(seq_id, None)
            return 0.0, 0.0, 0.0

        batch = self.runtime.build_tree_batch_from_histories(
            histories=[job.history for _, job in jobs],
            recovery_tokens=torch.tensor([job.recovery_token for _, job in jobs], dtype=torch.int64, device=self.device),
            temperatures=torch.tensor(
                [float(temperatures[row_idx].item()) for row_idx, _job in jobs],
                dtype=torch.float32,
                device=self.device,
            ),
            return_predicted_features=True,
        )

        per_seq_candidates: dict[int, tuple[int, dict[tuple[int, ...], PendingCandidateEntry]]] = {}
        for built_entry, (row_idx, job) in zip(batch.entries, jobs):
            candidate_map = per_seq_candidates.setdefault(
                seq_ids[row_idx],
                (job.frontier_version, {}),
            )[1]
            candidate_map[job.frontier_token_ids] = PendingCandidateEntry(
                entry=built_entry,
                rank=job.source_rank,
            )
        for seq_id in seq_ids:
            if seq_id in per_seq_candidates:
                self.pending_candidates[seq_id] = per_seq_candidates[seq_id]
            else:
                self.pending_candidates.pop(seq_id, None)
        return batch.dflash_time_s, batch.predictor_time_s, batch.tree_build_time_s

    def _handle_speculate(self) -> None:
        batch_size = int(self._recv_int64((1,)).item())
        request_meta = self._recv_int64((batch_size, 3))
        temperatures = self._recv_tensor((batch_size,), torch.float32)
        exact_feature_updates = self._recv_feature_list(batch_size)

        t0 = perf_counter()

        seq_ids = request_meta[:, 0].tolist()
        frontier_versions = request_meta[:, 1].tolist()
        recovery_tokens = request_meta[:, 2].tolist()

        self.runtime.commit_exact_context(seq_ids, frontier_versions, exact_feature_updates)

        cache_hits = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        entries: list[DDTreeEntry | None] = [None] * batch_size
        cache_lookup_s = 0.0
        for row_idx, (seq_id, frontier_version, recovery_token) in enumerate(zip(seq_ids, frontier_versions, recovery_tokens)):
            active = self.active_trees.get(seq_id)
            if active is None:
                continue
            if active.frontier_version != frontier_version or active.recovery_token != recovery_token:
                self.active_trees.pop(seq_id, None)
                continue
            cache_hits[row_idx] = 1
            entries[row_idx] = active.entry

        miss_indices = [idx for idx, entry in enumerate(entries) if entry is None]
        miss_entries, miss_batch = self._build_current_entries(
            miss_indices=miss_indices,
            seq_ids=seq_ids,
            frontier_versions=frontier_versions,
            recovery_tokens=recovery_tokens,
            temperatures=temperatures,
        )
        for row_idx, entry in miss_entries.items():
            entries[row_idx] = entry

        typed_entries = [entry for entry in entries if entry is not None]
        if len(typed_entries) != batch_size:
            raise RuntimeError("DDTree SSD runner failed to materialize all tree entries")

        service_dflash_s = 0.0 if miss_batch is None else miss_batch.dflash_time_s
        service_predictor_s = 0.0 if miss_batch is None else miss_batch.predictor_time_s
        service_tree_build_s = 0.0 if miss_batch is None else miss_batch.tree_build_time_s

        background_dflash_s, background_predictor_s, background_tree_build_s = self._populate_surrogate_candidates(
            seq_ids=seq_ids,
            frontier_versions=frontier_versions,
            temperatures=temperatures,
            entries=typed_entries,
        )
        for seq_id, frontier_version, temp, entry in zip(seq_ids, frontier_versions, temperatures.tolist(), typed_entries):
            self.pending_feedback[seq_id] = PendingDDTreeFeedback(
                frontier_version=frontier_version,
                temperature=float(temp),
                entry=entry,
            )

        num_nodes, node_token_ids, node_depths, parents = self._pack_entries(typed_entries)
        self._send_tensor(cache_hits)
        self._send_tensor(num_nodes)
        self._send_tensor(node_token_ids)
        self._send_tensor(node_depths)
        self._send_tensor(parents)
        self._send_speculate_diag(
            cache_lookup_s=cache_lookup_s,
            service_dflash_s=service_dflash_s,
            service_predictor_s=service_predictor_s,
            service_tree_build_s=service_tree_build_s,
            background_dflash_s=background_dflash_s,
            background_predictor_s=background_predictor_s,
            background_tree_build_s=background_tree_build_s,
            worker_total_s=perf_counter() - t0,
            cache_hits=cache_hits,
            entries=typed_entries,
        )

    def _handle_post_verify(self) -> None:
        batch_size = int(self._recv_int64((1,)).item())
        post_meta = self._recv_int64((batch_size, 3))
        suffixes = self._recv_suffixes(batch_size)
        exact_feature_updates = None
        if self.config.ddtree_context_mode == "exact" and self.config.ddtree_frontier_mode == "oracle":
            exact_feature_updates = self._recv_feature_list(batch_size)

        seq_ids = post_meta[:, 0].tolist()
        accepted_len_idxs = post_meta[:, 1].tolist()
        recovery_tokens = post_meta[:, 2].tolist()
        t0 = perf_counter()
        actual_frontier_rank = [-1] * batch_size
        background_dflash_s = 0.0
        background_predictor_s = 0.0
        background_tree_build_s = 0.0

        if self.config.ddtree_cache == "on" and self.config.ddtree_frontier_mode == "oracle":
            histories = []
            temps = []
            for row_idx, seq_id in enumerate(seq_ids):
                pending = self.pending_feedback.get(seq_id)
                if pending is None:
                    raise RuntimeError(f"Missing DDTree pending feedback for seq_id={seq_id}")
                temps.append(pending.temperature)
                if self.config.ddtree_context_mode == "exact":
                    history = self.runtime.build_exact_frontier_history(
                        seq_id=seq_id,
                        exact_feature_update=(exact_feature_updates or [None] * batch_size)[row_idx],
                    )
                else:
                    history = self.runtime.build_predicted_frontier_history(
                        seq_id=seq_id,
                        entry=pending.entry,
                        accepted_len_idx=accepted_len_idxs[row_idx],
                    )
                histories.append(history)
            batch = self.runtime.build_tree_batch_from_histories(
                histories=histories,
                recovery_tokens=torch.tensor(recovery_tokens, dtype=torch.int64, device=self.device),
                temperatures=torch.tensor(temps, dtype=torch.float32, device=self.device),
                return_predicted_features=(self.config.ddtree_context_mode == "predicted"),
            )
            background_dflash_s = batch.dflash_time_s
            background_predictor_s = batch.predictor_time_s
            background_tree_build_s = batch.tree_build_time_s
            for seq_id, pending, recovery_token, entry in zip(seq_ids, [self.pending_feedback[s] for s in seq_ids], recovery_tokens, batch.entries):
                self.active_trees[seq_id] = ActiveDDTree(
                    frontier_version=pending.frontier_version + 1,
                    recovery_token=int(recovery_token),
                    entry=entry,
                )
            actual_frontier_rank = [0] * batch_size
        elif self.config.ddtree_cache == "on" and self.config.ddtree_frontier_mode == "surrogate":
            for row_idx, (seq_id, suffix, recovery_token) in enumerate(zip(seq_ids, suffixes, recovery_tokens)):
                frontier = self.pending_candidates.get(seq_id)
                pending = self.pending_feedback.get(seq_id)
                if frontier is None or pending is None:
                    self.active_trees.pop(seq_id, None)
                    continue
                frontier_version, candidate_map = frontier
                actual_key = tuple(suffix + [int(recovery_token)])
                candidate = candidate_map.get(actual_key)
                if candidate is None or frontier_version != pending.frontier_version + 1:
                    self.active_trees.pop(seq_id, None)
                    continue
                actual_frontier_rank[row_idx] = int(candidate.rank)
                self.active_trees[seq_id] = ActiveDDTree(
                    frontier_version=frontier_version,
                    recovery_token=int(recovery_token),
                    entry=candidate.entry,
                )
        elif self.config.ddtree_context_mode == "predicted" and self.config.ddtree_frontier_mode == "oracle":
            entries = []
            frontier_versions = []
            for seq_id in seq_ids:
                pending = self.pending_feedback.get(seq_id)
                if pending is None:
                    raise RuntimeError(f"Missing DDTree pending feedback for seq_id={seq_id}")
                entries.append(pending.entry)
                frontier_versions.append(pending.frontier_version)
            self.runtime.store_oracle_predicted_frontiers(
                seq_ids=seq_ids,
                frontier_versions=frontier_versions,
                accepted_len_idxs=accepted_len_idxs,
                entries=entries,
            )
            actual_frontier_rank = [0] * batch_size

        for seq_id in seq_ids:
            self.pending_feedback.pop(seq_id, None)
            if self.config.ddtree_frontier_mode != "surrogate":
                self.pending_candidates.pop(seq_id, None)

        diag = torch.tensor(
            [
                background_dflash_s,
                background_predictor_s,
                background_tree_build_s,
                perf_counter() - t0,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self._send_tensor(diag)
        self._send_tensor(torch.tensor(actual_frontier_rank, dtype=torch.int64, device=self.device))

    def loop(self) -> None:
        while True:
            cmd = int(self._recv_int64((1,)).item())
            if cmd == DDTreeSSDCommand.PREFILL:
                self._handle_prefill()
            elif cmd == DDTreeSSDCommand.SPECULATE:
                self._handle_speculate()
            elif cmd == DDTreeSSDCommand.POST_VERIFY:
                self._handle_post_verify()
            elif cmd == DDTreeSSDCommand.EXIT:
                break
            else:
                raise RuntimeError(f"Unknown ddtree_ssd command: {cmd}")
