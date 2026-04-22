from time import perf_counter

import torch
import torch.distributed as dist

from ssd.engine.helpers.ddtree import DDTreeEntry, compile_verify_inputs
from ssd.engine.helpers.speculate_types import DDTreeDiagnosticBatch, SpeculateResult, VerifyResult, SpeculatorBase
from ssd.engine.sequence import Sequence


class DDTreeSSDCommand:
    SPECULATE = 0
    PREFILL = 1
    EXIT = 2
    POST_VERIFY = 3


class SpeculatorAsyncDDTree(SpeculatorBase):
    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        tree_budget: int,
        draft_dtype: torch.dtype,
        feature_dim: int,
        async_pg: dist.ProcessGroup,
        draft_runner_rank: int,
        context_mode: str,
        cache_mode: str,
        frontier_mode: str,
    ):
        super().__init__(lookahead, device)
        self.tree_budget = tree_budget
        self.draft_dtype = draft_dtype
        self.feature_dim = feature_dim
        self.async_pg = async_pg
        self.draft_runner_rank = draft_runner_rank
        self.context_mode = context_mode
        self.cache_mode = cache_mode
        self.frontier_mode = frontier_mode
        self._alloc_bufs(1)

    def _alloc_bufs(self, batch_size: int) -> None:
        self._buf_batch_size = batch_size
        self._cmd = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._meta = torch.empty(1, dtype=torch.int64, device=self.device)
        self._request_meta = torch.empty((batch_size, 3), dtype=torch.int64, device=self.device)
        self._temps = torch.empty((batch_size,), dtype=torch.float32, device=self.device)
        self._cache_hits = torch.empty((batch_size,), dtype=torch.int64, device=self.device)
        self._num_nodes = torch.empty((batch_size,), dtype=torch.int64, device=self.device)
        self._node_token_ids = torch.empty((batch_size, self.tree_budget), dtype=torch.int64, device=self.device)
        self._node_depths = torch.empty((batch_size, self.tree_budget), dtype=torch.int64, device=self.device)
        self._parents = torch.empty((batch_size, self.tree_budget), dtype=torch.int64, device=self.device)
        self._batch_diag = torch.empty((8,), dtype=torch.float32, device=self.device)
        self._row_diag = torch.empty((batch_size, 2), dtype=torch.int64, device=self.device)
        self._post_meta = torch.empty((batch_size, 3), dtype=torch.int64, device=self.device)
        self._post_diag = torch.empty((4,), dtype=torch.float32, device=self.device)
        self._post_rank = torch.empty((batch_size,), dtype=torch.int64, device=self.device)

    def _send_feature_list(self, feature_list: list[torch.Tensor | None]) -> None:
        lengths = [0 if feat is None else int(feat.shape[0]) for feat in feature_list]
        dist.send(
            torch.tensor(lengths, dtype=torch.int64, device=self.device),
            dst=self.draft_runner_rank,
            group=self.async_pg,
        )
        total = sum(lengths)
        if total == 0:
            return
        flat = torch.cat(
            [
                feat.to(device=self.device, dtype=self.draft_dtype, non_blocking=False)
                for feat in feature_list
                if feat is not None and feat.numel() > 0
            ],
            dim=0,
        )
        dist.send(flat, dst=self.draft_runner_rank, group=self.async_pg)

    def _send_suffixes(self, suffixes: list[list[int]]) -> None:
        lengths = torch.tensor([len(suffix) for suffix in suffixes], dtype=torch.int64, device=self.device)
        dist.send(lengths, dst=self.draft_runner_rank, group=self.async_pg)
        total = int(lengths.sum().item())
        if total == 0:
            return
        flat = torch.tensor(
            [token for suffix in suffixes for token in suffix],
            dtype=torch.int64,
            device=self.device,
        )
        dist.send(flat, dst=self.draft_runner_rank, group=self.async_pg)

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        if verify_result.dflash_target_features is None:
            raise RuntimeError("ddtree_ssd prefill requires prompt DFlash target features")
        batch_size = len(seqs)
        if batch_size != self._buf_batch_size:
            self._alloc_bufs(batch_size)
        for seq in seqs:
            seq.frontier_version = 0
            seq.extend_dflash_target_features = None
            seq.extend_dflash_token_ids = None
            seq.extend_dflash_count = 0
            seq.dflash_cycle_idx = 0
        self._cmd.fill_(DDTreeSSDCommand.PREFILL)
        self._meta[0] = batch_size
        seq_ids = torch.tensor([seq.seq_id for seq in seqs], dtype=torch.int64, device=self.device)
        dist.send(self._cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._meta, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(seq_ids, dst=self.draft_runner_rank, group=self.async_pg)
        self._send_feature_list(verify_result.dflash_target_features)
        return SpeculateResult(torch.empty(0), torch.empty(0))

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        batch_size = len(seqs)
        if batch_size != self._buf_batch_size:
            self._alloc_bufs(batch_size)
        exact_feature_updates = []
        for row_idx, seq in enumerate(seqs):
            if seq.recovery_token_id is None:
                raise ValueError(f"recovery_token_id is None for seq_id={seq.seq_id}")
            self._request_meta[row_idx, 0] = seq.seq_id
            self._request_meta[row_idx, 1] = seq.frontier_version
            self._request_meta[row_idx, 2] = seq.recovery_token_id
            self._temps[row_idx] = seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            exact_feature_updates.append(seq.extend_dflash_target_features)
            seq.append_token(seq.recovery_token_id)

        self._cmd.fill_(DDTreeSSDCommand.SPECULATE)
        self._meta[0] = batch_size
        t0 = perf_counter()
        dist.send(self._cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._meta, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._request_meta, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._temps, dst=self.draft_runner_rank, group=self.async_pg)
        self._send_feature_list(exact_feature_updates)

        dist.recv(self._cache_hits, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._num_nodes, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._node_token_ids, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._node_depths, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._parents, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._batch_diag, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._row_diag, src=self.draft_runner_rank, group=self.async_pg)
        speculate_wait_s = perf_counter() - t0

        entries = []
        for row_idx, seq in enumerate(seqs):
            entry = DDTreeEntry(
                recovery_token=int(self._request_meta[row_idx, 2].item()),
                node_token_ids=self._node_token_ids[row_idx].clone(),
                node_depths=self._node_depths[row_idx].clone(),
                parents=self._parents[row_idx].clone(),
                num_nodes=int(self._num_nodes[row_idx].item()),
                draft_tokens=torch.empty(0, dtype=torch.int64, device=self.device),
                logits_q=torch.empty(0, dtype=self.draft_dtype, device=self.device),
                branch_logits=torch.empty(0, dtype=self.draft_dtype, device=self.device),
            )
            entry.verify_input_ids, entry.verify_positions = compile_verify_inputs(
                recovery_token=entry.recovery_token,
                node_token_ids=entry.node_token_ids,
                node_depths=entry.node_depths,
                num_nodes=entry.num_nodes,
                prefix_len=seq.num_cached_tokens,
                device=self.device,
            )
            entries.append(entry)
        diag = DDTreeDiagnosticBatch(
            cache_lookup_s=float(self._batch_diag[0].item()),
            service_dflash_s=float(self._batch_diag[1].item()),
            service_predictor_s=float(self._batch_diag[2].item()),
            service_tree_build_s=float(self._batch_diag[3].item()),
            background_dflash_s=float(self._batch_diag[4].item()),
            background_predictor_s=float(self._batch_diag[5].item()),
            background_tree_build_s=float(self._batch_diag[6].item()),
            worker_total_s=float(self._batch_diag[7].item()),
            speculate_wait_s=speculate_wait_s,
            fallback_used=[bool(x) for x in self._row_diag[:, 0].tolist()],
            frontier_candidate_count=[int(x) for x in self._row_diag[:, 1].tolist()],
        )
        return SpeculateResult(
            speculations=torch.empty(0, dtype=torch.int64, device=self.device),
            logits_q=torch.empty(0, device=self.device),
            cache_hits=self._cache_hits.clone(),
            ddtree_diag=diag,
            ddtree_entries=entries,
        )

    def post_verify_feedback(self, seqs: list[Sequence], verify_result: VerifyResult, diag: DDTreeDiagnosticBatch | None = None) -> None:
        batch_size = len(seqs)
        if batch_size == 0:
            return
        if batch_size != self._buf_batch_size:
            self._alloc_bufs(batch_size)
        for row_idx, (seq, suffix, recovery_token) in enumerate(
            zip(seqs, verify_result.new_suffixes, verify_result.recovery_tokens)
        ):
            self._post_meta[row_idx, 0] = seq.seq_id
            self._post_meta[row_idx, 1] = len(suffix) - 1
            self._post_meta[row_idx, 2] = recovery_token

        self._cmd.fill_(DDTreeSSDCommand.POST_VERIFY)
        self._meta[0] = batch_size
        t0 = perf_counter()
        dist.send(self._cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._meta, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._post_meta, dst=self.draft_runner_rank, group=self.async_pg)
        self._send_suffixes(verify_result.new_suffixes)
        if self.context_mode == "exact" and self.frontier_mode == "oracle" and verify_result.dflash_target_features is not None:
            self._send_feature_list(verify_result.dflash_target_features)
        dist.recv(self._post_diag, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._post_rank, src=self.draft_runner_rank, group=self.async_pg)
        if diag is not None:
            diag.background_dflash_s = float(self._post_diag[0].item())
            diag.background_predictor_s = float(self._post_diag[1].item())
            diag.background_tree_build_s = float(self._post_diag[2].item())
            diag.worker_total_s += float(self._post_diag[3].item())
            diag.post_verify_wait_s = perf_counter() - t0
            diag.actual_frontier_rank = [
                None if value < 0 else int(value)
                for value in self._post_rank.tolist()
            ]
