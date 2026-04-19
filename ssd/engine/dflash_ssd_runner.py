import os
from dataclasses import dataclass
from time import perf_counter

import torch
import torch.distributed as dist

from ssd.config import Config
from ssd.engine.dflash_runtime import DFlashCacheEntry, DFlashRuntime


class DFlashSSDCommand:
    SPECULATE = 0
    PREFILL = 1
    EXIT = 2
    POST_VERIFY = 3


@dataclass
class PendingPostVerifyBatch:
    seq_ids: list[int]
    frontier_versions: list[int]
    temperatures: torch.Tensor
    entries: list[DFlashCacheEntry]
    cache_hits: torch.Tensor
    recovery_tokens: list[int]


class DFlashSSDRunner:
    def __init__(self, config: Config, rank: int):
        self.config = config
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.dtype = getattr(config.draft_hf_config, "torch_dtype", None) or torch.bfloat16
        self.lookahead = config.speculate_k
        self.feature_dim = config.dflash_target_feature_dim
        self.tree_cache: dict[tuple[int, int, int, int], DFlashCacheEntry] = {}
        self.pending_post_verify: PendingPostVerifyBatch | None = None
        self.metrics = {
            "dflash_draft_step_times": [],
            "dflash_predictor_times": [],
        }

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

        self.runtime = DFlashRuntime(
            config=config,
            device=self.device,
            predictor_path=config.dflash_predictor,
            metrics=self.metrics,
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

    def _handle_prefill(self) -> None:
        batch_size = int(self._recv_int64((1,)).item())
        seq_ids = self._recv_int64((batch_size,)).tolist()
        prompt_target_features = self._recv_feature_list(batch_size)
        self.runtime.prefill_exact_context(seq_ids, prompt_target_features, frontier_version=0)
        self.pending_post_verify = None

    def _lookup_cache(
        self,
        cache_keys: torch.Tensor,
    ) -> tuple[torch.Tensor, list[DFlashCacheEntry | None], list[int | None], float]:
        batch_size = cache_keys.shape[0]
        cache_hits = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        entries: list[DFlashCacheEntry | None] = [None] * batch_size
        true_branch_ranks: list[int | None] = [None] * batch_size
        if self.config.dflash_branch_cache != "on":
            return cache_hits, entries, true_branch_ranks, 0.0

        t0 = perf_counter()
        for row_idx, key in enumerate(cache_keys.tolist()):
            entry = self.tree_cache.get(tuple(key))
            if entry is None:
                continue
            cache_hits[row_idx] = 1
            entries[row_idx] = entry
            true_branch_ranks[row_idx] = entry.branch_rank
        return cache_hits, entries, true_branch_ranks, perf_counter() - t0

    def _materialize_entries(
        self,
        miss_indices: list[int],
        seq_ids: list[int],
        frontier_versions: list[int],
        recovery_tokens: list[int],
        temperatures: torch.Tensor,
    ) -> tuple[dict[int, DFlashCacheEntry], list[bool], float, float]:
        miss_entries: dict[int, DFlashCacheEntry] = {}
        fallback_used = [False] * len(seq_ids)
        if not miss_indices:
            return miss_entries, fallback_used, 0.0, 0.0

        histories: list[torch.Tensor] = []
        miss_seq_ids: list[int] = []
        rec_tokens: list[int] = []
        temps: list[float] = []
        miss_used_predicted: list[bool] = []

        prefer_predicted = (
            self.config.dflash_context_mode == "predicted"
            and self.config.dflash_branch_cache == "off"
            and self.config.dflash_branch_key_mode == "oracle"
        )

        for req_idx in miss_indices:
            if prefer_predicted:
                history, used_predicted = self.runtime.get_generation_history(
                    seq_ids[req_idx],
                    frontier_versions[req_idx],
                    prefer_predicted=True,
                )
                fallback_used[req_idx] = not used_predicted
                miss_used_predicted.append(used_predicted)
            else:
                history = self.runtime.get_exact_history(seq_ids[req_idx]).clone()
                fallback_used[req_idx] = True
                miss_used_predicted.append(False)

            histories.append(history)
            miss_seq_ids.append(seq_ids[req_idx])
            rec_tokens.append(recovery_tokens[req_idx])
            temps.append(float(temperatures[req_idx].item()))

        outputs = self.runtime.run_block_batch(
            histories,
            torch.tensor(rec_tokens, dtype=torch.int64, device=self.device),
            torch.tensor(temps, dtype=torch.float32, device=self.device),
            return_predicted_features=(self.config.dflash_context_mode == "predicted"),
        )
        for out_row, req_idx in enumerate(miss_indices):
            miss_entries[req_idx] = DFlashCacheEntry(
                tokens=outputs.draft_tokens[out_row].clone(),
                logits_q=outputs.logits_q[out_row].clone(),
                branch_logits=outputs.branch_logits[out_row].clone(),
                predicted_target_features=(
                    outputs.predicted_target_features[out_row].clone()
                    if outputs.predicted_target_features is not None else None
                ),
            )
        return miss_entries, fallback_used, outputs.dflash_time_s, outputs.predictor_time_s

    def _send_response(
        self,
        cache_hits: torch.Tensor,
        entries: list[DFlashCacheEntry],
        fallback_used: list[bool],
        true_branch_ranks: list[int | None],
        num_branches_generated: list[int],
        cache_lookup_s: float,
        service_dflash_s: float,
        service_predictor_s: float,
        background_dflash_s: float,
        background_predictor_s: float,
        worker_total_s: float,
    ) -> None:
        out_tokens = torch.stack([entry.tokens for entry in entries], dim=0)
        out_logits = torch.stack([entry.logits_q for entry in entries], dim=0)
        fused_response = torch.cat([cache_hits, out_tokens.reshape(-1).to(torch.int64)], dim=0)
        self._send_tensor(fused_response)
        self._send_tensor(out_logits)

        batch_diag = torch.tensor(
            [
                cache_lookup_s,
                service_dflash_s,
                service_predictor_s,
                background_dflash_s,
                background_predictor_s,
                worker_total_s,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        row_diag = torch.tensor(
            [
                [
                    int(fallback_used[row_idx]),
                    -1 if true_branch_ranks[row_idx] is None else int(true_branch_ranks[row_idx]),
                    int(num_branches_generated[row_idx]),
                ]
                for row_idx in range(len(entries))
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._send_tensor(batch_diag)
        self._send_tensor(row_diag)

    def _invalidate_seq_cache(self, seq_ids: list[int]) -> None:
        seq_id_set = set(seq_ids)
        self.tree_cache = {k: v for k, v in self.tree_cache.items() if k[0] not in seq_id_set}

    def _handle_speculate(self) -> None:
        batch_size = int(self._recv_int64((1,)).item())
        cache_keys = self._recv_int64((batch_size, 4))
        temperatures = self._recv_tensor((batch_size,), torch.float32)
        exact_feature_updates = self._recv_feature_list(batch_size)

        t0 = perf_counter()
        seq_ids = cache_keys[:, 0].tolist()
        frontier_versions = cache_keys[:, 1].tolist()
        recovery_tokens = cache_keys[:, 3].tolist()
        self.runtime.commit_exact_context(seq_ids, frontier_versions, exact_feature_updates)

        cache_hits, entries, true_branch_ranks, cache_lookup_s = self._lookup_cache(cache_keys)
        miss_indices = [idx for idx, entry in enumerate(entries) if entry is None]
        miss_entries, fallback_used, service_dflash_s, service_predictor_s = self._materialize_entries(
            miss_indices,
            seq_ids,
            frontier_versions,
            recovery_tokens,
            temperatures,
        )
        for row_idx, entry in miss_entries.items():
            entries[row_idx] = entry

        if any(entry is None for entry in entries):
            raise RuntimeError("DFlash SSD runner failed to materialize all request entries")
        typed_entries = [entry for entry in entries if entry is not None]

        background_dflash_s = 0.0
        background_predictor_s = 0.0
        num_branches_generated = [0] * batch_size
        self.pending_post_verify = None
        feedback_needed = self.config.dflash_enable_diagnostics or self.config.dflash_branch_key_mode == "oracle"

        if self.config.dflash_context_mode == "predicted":
            if self.config.dflash_branch_cache == "on" and self.config.dflash_branch_key_mode == "normal":
                populate = self.runtime.populate_branch_cache(
                    seq_ids=seq_ids,
                    frontier_versions=frontier_versions,
                    recovery_tokens=recovery_tokens,
                    temperatures=temperatures,
                    cache_hits=cache_hits,
                    entries=typed_entries,
                )
                background_dflash_s = populate.dflash_time_s
                background_predictor_s = populate.predictor_time_s
                num_branches_generated = populate.num_branches_generated
                self._invalidate_seq_cache(seq_ids)
                self.tree_cache.update(populate.cache)
            if feedback_needed:
                self.pending_post_verify = PendingPostVerifyBatch(
                    seq_ids=seq_ids,
                    frontier_versions=frontier_versions,
                    temperatures=temperatures.clone(),
                    entries=[entry for entry in typed_entries],
                    cache_hits=cache_hits.clone(),
                    recovery_tokens=list(recovery_tokens),
                )
            else:
                if self.config.dflash_branch_cache != "on":
                    self._invalidate_seq_cache(seq_ids)
        else:
            if self.config.dflash_branch_cache == "on" and self.config.dflash_branch_key_mode == "oracle":
                self.pending_post_verify = PendingPostVerifyBatch(
                    seq_ids=seq_ids,
                    frontier_versions=frontier_versions,
                    temperatures=temperatures.clone(),
                    entries=[entry for entry in typed_entries],
                    cache_hits=cache_hits.clone(),
                    recovery_tokens=list(recovery_tokens),
                )
            else:
                self._invalidate_seq_cache(seq_ids)

        self._send_response(
            cache_hits=cache_hits,
            entries=typed_entries,
            fallback_used=fallback_used,
            true_branch_ranks=true_branch_ranks,
            num_branches_generated=num_branches_generated,
            cache_lookup_s=cache_lookup_s,
            service_dflash_s=service_dflash_s,
            service_predictor_s=service_predictor_s,
            background_dflash_s=background_dflash_s,
            background_predictor_s=background_predictor_s,
            worker_total_s=perf_counter() - t0,
        )

    def _handle_post_verify(self) -> None:
        batch_size = int(self._recv_int64((1,)).item())
        post_verify_meta = self._recv_int64((batch_size, 3))
        seq_ids = post_verify_meta[:, 0].tolist()
        accepted_len_idxs = post_verify_meta[:, 1].tolist()
        recovery_tokens = post_verify_meta[:, 2].tolist()
        exact_feature_updates = None
        if self.config.dflash_context_mode == "exact" and self.config.dflash_branch_key_mode == "oracle":
            exact_feature_updates = self._recv_feature_list(batch_size)

        t0 = perf_counter()
        num_branches_generated = [0] * batch_size
        background_dflash_s = 0.0
        background_predictor_s = 0.0
        actual_accept_supported = [-1] * batch_size
        actual_recovery_rank_given_accept = [-1] * batch_size
        joint_branch_supported = [-1] * batch_size
        branch_score_stats = torch.full((batch_size, 2), -1.0, dtype=torch.float32, device=self.device)

        if self.pending_post_verify is not None:
            pending_map = {
                seq_id: (
                    frontier_version,
                    self.pending_post_verify.temperatures[row_idx],
                    self.pending_post_verify.entries[row_idx],
                    bool(self.pending_post_verify.cache_hits[row_idx].item()),
                    self.pending_post_verify.recovery_tokens[row_idx],
                )
                for row_idx, (seq_id, frontier_version) in enumerate(
                    zip(self.pending_post_verify.seq_ids, self.pending_post_verify.frontier_versions)
                )
            }
            frontier_versions: list[int] = []
            temperatures: list[float] = []
            entries: list[DFlashCacheEntry] = []
            cache_hits: list[int] = []
            request_recovery_tokens: list[int] = []
            for seq_id in seq_ids:
                if seq_id not in pending_map:
                    raise RuntimeError(f"Missing pending post-verify state for seq_id={seq_id}")
                frontier_version, temp, entry, cache_hit, request_recovery_token = pending_map[seq_id]
                frontier_versions.append(frontier_version)
                temperatures.append(float(temp.item()))
                entries.append(entry)
                cache_hits.append(int(cache_hit))
                request_recovery_tokens.append(int(request_recovery_token))

            if self.config.dflash_branch_cache == "on":
                branch_diag = self.runtime.analyze_realized_branches(
                    recovery_tokens=request_recovery_tokens,
                    cache_hits=torch.tensor(cache_hits, dtype=torch.int64, device=self.device),
                    accepted_len_idxs=accepted_len_idxs,
                    realized_recovery_tokens=recovery_tokens,
                    entries=entries,
                )
                actual_accept_supported = [1 if value else 0 for value in branch_diag.actual_accept_supported]
                actual_recovery_rank_given_accept = [
                    -1 if value is None else int(value)
                    for value in branch_diag.actual_recovery_rank_given_accept
                ]
                joint_branch_supported = [1 if value else 0 for value in branch_diag.joint_branch_supported]
                for row_idx in range(batch_size):
                    entropy = branch_diag.recovery_entropy_at_actual_accept[row_idx]
                    margin = branch_diag.recovery_top1_margin_at_actual_accept[row_idx]
                    branch_score_stats[row_idx, 0] = -1.0 if entropy is None else float(entropy)
                    branch_score_stats[row_idx, 1] = -1.0 if margin is None else float(margin)

            if self.config.dflash_branch_key_mode == "oracle":
                if self.config.dflash_branch_cache == "on":
                    if self.config.dflash_context_mode == "exact":
                        populate = self.runtime.populate_exact_oracle_branch_cache(
                            seq_ids=seq_ids,
                            frontier_versions=frontier_versions,
                            accepted_len_idxs=accepted_len_idxs,
                            recovery_tokens=recovery_tokens,
                            temperatures=torch.tensor(temperatures, dtype=torch.float32, device=self.device),
                            exact_feature_updates=exact_feature_updates or [None] * batch_size,
                        )
                    else:
                        populate = self.runtime.populate_oracle_branch_cache(
                            seq_ids=seq_ids,
                            frontier_versions=frontier_versions,
                            accepted_len_idxs=accepted_len_idxs,
                            recovery_tokens=recovery_tokens,
                            temperatures=torch.tensor(temperatures, dtype=torch.float32, device=self.device),
                            entries=entries,
                        )
                    background_dflash_s = populate.dflash_time_s
                    background_predictor_s = populate.predictor_time_s
                    num_branches_generated = populate.num_branches_generated
                    self._invalidate_seq_cache(seq_ids)
                    self.tree_cache.update(populate.cache)
                elif self.config.dflash_context_mode == "predicted":
                    self.runtime.store_oracle_predicted_frontiers(
                        seq_ids=seq_ids,
                        frontier_versions=frontier_versions,
                        accepted_len_idxs=accepted_len_idxs,
                        entries=entries,
                    )

        self.pending_post_verify = None
        batch_diag = torch.tensor(
            [
                background_dflash_s,
                background_predictor_s,
                perf_counter() - t0,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        row_counts = torch.tensor(num_branches_generated, dtype=torch.int64, device=self.device)
        row_branch_diag = torch.tensor(
            [
                [actual_accept_supported[row_idx], actual_recovery_rank_given_accept[row_idx], joint_branch_supported[row_idx]]
                for row_idx in range(batch_size)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._send_tensor(batch_diag)
        self._send_tensor(row_counts)
        self._send_tensor(row_branch_diag)
        self._send_tensor(branch_score_stats)

    def loop(self) -> None:
        while True:
            cmd = int(self._recv_int64((1,)).item())
            if cmd == DFlashSSDCommand.PREFILL:
                self._handle_prefill()
            elif cmd == DFlashSSDCommand.SPECULATE:
                self._handle_speculate()
            elif cmd == DFlashSSDCommand.POST_VERIFY:
                self._handle_post_verify()
            elif cmd == DFlashSSDCommand.EXIT:
                break
            else:
                raise RuntimeError(f"Unknown dflash_ssd command: {cmd}")
