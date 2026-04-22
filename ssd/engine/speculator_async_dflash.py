from time import perf_counter

import torch
import torch.distributed as dist

from ssd.engine.helpers.speculate_types import (
    DFlashDiagnosticBatch,
    SpeculateResult,
    VerifyResult,
    SpeculatorBase,
)
from ssd.engine.sequence import Sequence


class DFlashSSDCommand:
    SPECULATE = 0
    PREFILL = 1
    EXIT = 2
    POST_VERIFY = 3


class SpeculatorAsyncDFlash(SpeculatorBase):
    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        vocab_size: int,
        draft_dtype: torch.dtype,
        feature_dim: int,
        async_pg: dist.ProcessGroup,
        draft_runner_rank: int,
        verbose: bool,
        branch_key_mode: str,
        context_mode: str,
    ):
        super().__init__(lookahead, device)
        self.vocab_size = vocab_size
        self.draft_dtype = draft_dtype
        self.feature_dim = feature_dim
        self.async_pg = async_pg
        self.draft_runner_rank = draft_runner_rank
        self.verbose = verbose
        self.branch_key_mode = branch_key_mode
        self.context_mode = context_mode
        self._alloc_bufs(1)

    def _alloc_bufs(self, batch_size: int) -> None:
        self._buf_batch_size = batch_size
        self._cmd = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._meta = torch.empty(1, dtype=torch.int64, device=self.device)
        self._cache_keys = torch.empty(batch_size, 4, dtype=torch.int64, device=self.device)
        self._temps_buf = torch.empty(batch_size, dtype=torch.float32, device=self.device)
        self._response = torch.empty(batch_size + batch_size * self.lookahead, dtype=torch.int64, device=self.device)
        self._logits_q = torch.empty(
            batch_size,
            self.lookahead,
            self.vocab_size,
            dtype=self.draft_dtype,
            device=self.device,
        )
        self._batch_diag = torch.empty(6, dtype=torch.float32, device=self.device)
        self._row_diag = torch.empty(batch_size, 3, dtype=torch.int64, device=self.device)
        self._oracle_meta = torch.empty(batch_size, 3, dtype=torch.int64, device=self.device)
        self._oracle_diag = torch.empty(3, dtype=torch.float32, device=self.device)
        self._oracle_counts = torch.empty(batch_size, dtype=torch.int64, device=self.device)
        self._oracle_branch_diag = torch.empty(batch_size, 3, dtype=torch.int64, device=self.device)
        self._oracle_branch_stats = torch.empty(batch_size, 2, dtype=torch.float32, device=self.device)
        self._speculations = torch.empty(batch_size, self.lookahead + 1, dtype=torch.int64, device=self.device)

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

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        if verify_result.dflash_target_features is None:
            raise RuntimeError("dflash_ssd async prefill requires prompt DFlash target features")
        batch_size = len(seqs)
        if batch_size != self._buf_batch_size:
            self._alloc_bufs(batch_size)
        for seq in seqs:
            seq.frontier_version = 0
            seq.extend_dflash_target_features = None
            seq.extend_dflash_token_ids = None
            seq.extend_dflash_count = 0
            seq.dflash_cycle_idx = 0
        self._cmd.fill_(DFlashSSDCommand.PREFILL)
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
        for i, seq in enumerate(seqs):
            if seq.recovery_token_id is None:
                raise ValueError(f"recovery_token_id is None for seq {i}")
            self._cache_keys[i, 0] = seq.seq_id
            self._cache_keys[i, 1] = seq.frontier_version
            self._cache_keys[i, 2] = seq.last_spec_step_accepted_len - 1
            self._cache_keys[i, 3] = seq.recovery_token_id
            self._temps_buf[i] = seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            exact_feature_updates.append(seq.extend_dflash_target_features)
            seq.append_token(seq.recovery_token_id)

        self._cmd.fill_(DFlashSSDCommand.SPECULATE)
        self._meta[0] = batch_size
        t0 = perf_counter()
        dist.send(self._cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._meta, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._cache_keys, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._temps_buf, dst=self.draft_runner_rank, group=self.async_pg)
        self._send_feature_list(exact_feature_updates)

        dist.recv(self._response, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._logits_q, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._batch_diag, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._row_diag, src=self.draft_runner_rank, group=self.async_pg)
        speculate_wait_s = perf_counter() - t0

        cache_hits = self._response[:batch_size]
        speculated_tokens = self._response[batch_size:].view(batch_size, self.lookahead)
        self._speculations[:, 0] = self._cache_keys[:, 3]
        self._speculations[:, 1:] = speculated_tokens

        for i, seq in enumerate(seqs):
            seq.token_ids.extend(speculated_tokens[i].tolist())
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
            seq.num_draft_cached_tokens += self.lookahead + 1

        diag = DFlashDiagnosticBatch(
            cache_lookup_s=float(self._batch_diag[0].item()),
            service_dflash_s=float(self._batch_diag[1].item()),
            service_predictor_s=float(self._batch_diag[2].item()),
            background_dflash_s=float(self._batch_diag[3].item()),
            background_predictor_s=float(self._batch_diag[4].item()),
            worker_total_s=float(self._batch_diag[5].item()),
            speculate_wait_s=speculate_wait_s,
            fallback_used=[bool(x) for x in self._row_diag[:, 0].tolist()],
            true_branch_rank=[
                None if value < 0 else int(value)
                for value in self._row_diag[:, 1].tolist()
            ],
            num_branches_generated=[int(x) for x in self._row_diag[:, 2].tolist()],
        )
        return SpeculateResult(self._speculations, self._logits_q, cache_hits, dflash_diag=diag)

    def post_verify_feedback(self, seqs: list[Sequence], verify_result: VerifyResult, diag: DFlashDiagnosticBatch) -> None:
        batch_size = len(seqs)
        if batch_size == 0:
            return
        if batch_size != self._buf_batch_size:
            self._alloc_bufs(batch_size)

        for row_idx, (seq, new_suffix, recovery_token) in enumerate(
            zip(seqs, verify_result.new_suffixes, verify_result.recovery_tokens)
        ):
            self._oracle_meta[row_idx, 0] = seq.seq_id
            self._oracle_meta[row_idx, 1] = len(new_suffix) - 1
            self._oracle_meta[row_idx, 2] = recovery_token

        self._cmd.fill_(DFlashSSDCommand.POST_VERIFY)
        self._meta[0] = batch_size
        t0 = perf_counter()
        dist.send(self._cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._meta, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._oracle_meta, dst=self.draft_runner_rank, group=self.async_pg)
        if self.context_mode == "exact" and self.branch_key_mode == "oracle" and verify_result.dflash_target_features is not None:
            self._send_feature_list(verify_result.dflash_target_features)
        dist.recv(self._oracle_diag, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._oracle_counts, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._oracle_branch_diag, src=self.draft_runner_rank, group=self.async_pg)
        dist.recv(self._oracle_branch_stats, src=self.draft_runner_rank, group=self.async_pg)
        if self.branch_key_mode == "oracle":
            diag.background_dflash_s = float(self._oracle_diag[0].item())
            diag.background_predictor_s = float(self._oracle_diag[1].item())
        diag.worker_total_s += float(self._oracle_diag[2].item())
        diag.post_verify_wait_s = perf_counter() - t0
        if self.branch_key_mode == "oracle":
            diag.num_branches_generated = [int(x) for x in self._oracle_counts.tolist()]
        diag.actual_accept_supported = (
            [bool(value) for value in self._oracle_branch_diag[:, 0].tolist()]
            if int((self._oracle_branch_diag[:, 0] >= 0).any().item()) else None
        )
        diag.actual_recovery_rank_given_accept = [
            None if value < 0 else int(value)
            for value in self._oracle_branch_diag[:, 1].tolist()
        ] if int((self._oracle_branch_diag[:, 0] >= 0).any().item()) else None
        diag.joint_branch_supported = (
            [bool(value) for value in self._oracle_branch_diag[:, 2].tolist()]
            if int((self._oracle_branch_diag[:, 2] >= 0).any().item()) else None
        )
        if int((self._oracle_branch_stats[:, 0] >= 0).any().item()):
            diag.recovery_entropy_at_actual_accept = [
                None if value < 0 else float(value)
                for value in self._oracle_branch_stats[:, 0].tolist()
            ]
            diag.recovery_top1_margin_at_actual_accept = [
                None if value < 0 else float(value)
                for value in self._oracle_branch_stats[:, 1].tolist()
            ]
        else:
            diag.recovery_entropy_at_actual_accept = None
            diag.recovery_top1_margin_at_actual_accept = None
