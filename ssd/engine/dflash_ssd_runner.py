import os
from time import perf_counter

import torch
import torch.distributed as dist

from ssd.config import Config
from ssd.engine.dflash_runtime import DFlashCacheEntry, DFlashRuntime


class DFlashSSDRunner:
    def __init__(self, config: Config, rank: int):
        self.config = config
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.dtype = getattr(config.draft_hf_config, "torch_dtype", None) or torch.bfloat16
        self.lookahead = config.speculate_k
        self.feature_dim = config.dflash_target_feature_dim
        self.tree_cache: dict[tuple[int, int, int, int], DFlashCacheEntry] = {}
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

    def _run_exact_misses(
        self,
        miss_indices: list[int],
        seq_ids: list[int],
        recovery_tokens: list[int],
        temperatures: torch.Tensor,
    ) -> dict[int, DFlashCacheEntry]:
        miss_entries: dict[int, DFlashCacheEntry] = {}
        if not miss_indices:
            return miss_entries

        miss_seq_ids = [seq_ids[idx] for idx in miss_indices]
        rec_tokens = torch.tensor([recovery_tokens[idx] for idx in miss_indices], dtype=torch.int64, device=self.device)
        temps = temperatures[miss_indices]
        outputs = self.runtime.generate_block(
            seq_ids=miss_seq_ids,
            recovery_tokens=rec_tokens,
            temperatures=temps,
            return_predicted_features=True,
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
        return miss_entries

    def _handle_speculate(self) -> None:
        self.metrics["dflash_predictor_times"].clear()
        t0 = perf_counter()
        batch_size = int(self._recv_int64((1,)).item())
        cache_keys = self._recv_int64((batch_size, 4))
        temperatures = self._recv_tensor((batch_size,), torch.float32)
        exact_feature_updates = self._recv_feature_list(batch_size)

        seq_ids = cache_keys[:, 0].tolist()
        frontier_versions = cache_keys[:, 1].tolist()
        recovery_tokens = cache_keys[:, 3].tolist()
        self.runtime.commit_exact_context(seq_ids, frontier_versions, exact_feature_updates)

        cache_hits = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        entries: list[DFlashCacheEntry | None] = [None] * batch_size
        miss_indices: list[int] = []
        for row_idx, key in enumerate(cache_keys.tolist()):
            entry = self.tree_cache.get(tuple(key))
            if entry is None:
                miss_indices.append(row_idx)
            else:
                cache_hits[row_idx] = 1
                entries[row_idx] = entry

        miss_entries = self._run_exact_misses(miss_indices, seq_ids, recovery_tokens, temperatures)
        for row_idx, entry in miss_entries.items():
            entries[row_idx] = entry

        if any(entry is None for entry in entries):
            raise RuntimeError("DFlash SSD runner failed to materialize all request entries")
        typed_entries = [entry for entry in entries if entry is not None]

        out_tokens = torch.stack([entry.tokens for entry in typed_entries], dim=0)
        out_logits = torch.stack([entry.logits_q for entry in typed_entries], dim=0)
        fused_response = torch.cat([cache_hits, out_tokens.reshape(-1).to(torch.int64)], dim=0)
        dist.send(fused_response, dst=0, group=self.async_pg)
        dist.send(out_logits.contiguous(), dst=0, group=self.async_pg)

        next_cache = self.runtime.populate_branch_cache(
            seq_ids=seq_ids,
            frontier_versions=frontier_versions,
            recovery_tokens=recovery_tokens,
            temperatures=temperatures,
            cache_hits=cache_hits,
            entries=typed_entries,
        )
        seq_id_set = set(seq_ids)
        self.tree_cache = {k: v for k, v in self.tree_cache.items() if k[0] not in seq_id_set}
        self.tree_cache.update(next_cache)

        timings = torch.tensor(
            [
                perf_counter() - t0,
                sum(self.metrics["dflash_predictor_times"]),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        dist.send(timings, dst=0, group=self.async_pg)

    def loop(self) -> None:
        while True:
            cmd = int(self._recv_int64((1,)).item())
            if cmd == 1:
                self._handle_prefill()
            elif cmd == 0:
                self._handle_speculate()
            elif cmd == 2:
                break
            else:
                raise RuntimeError(f"Unknown dflash_ssd command: {cmd}")
