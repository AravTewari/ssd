import torch

from ssd.engine.ddtree_worker import DDTreeWorkerHandle
from ssd.engine.helpers.ddtree import compile_verify_inputs
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase
from ssd.engine.sequence import Sequence


class SpeculatorSyncDDTree(SpeculatorBase):
    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        ddtree_worker: DDTreeWorkerHandle,
    ):
        super().__init__(lookahead, device)
        self.ddtree_worker = ddtree_worker

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        if verify_result.dflash_target_features is None:
            raise RuntimeError("DDTree prefill requires prompt target features from the verifier")
        for seq, prompt_features in zip(seqs, verify_result.dflash_target_features):
            seq.frontier_version = 0
            seq.extend_dflash_target_features = None
            seq.extend_dflash_token_ids = None
            seq.extend_dflash_count = 0
            seq.dflash_cycle_idx = 0
            seq.last_dflash_target_feature = prompt_features[-1].clone()
        self.ddtree_worker.prefill(
            seq_ids=[seq.seq_id for seq in seqs],
            prompt_target_features=verify_result.dflash_target_features,
        )
        return SpeculateResult(torch.empty(0), torch.empty(0))

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        seq_ids = []
        frontier_versions = []
        recovery_tokens = []
        temperatures = []
        target_features = []
        for seq in seqs:
            if seq.recovery_token_id is None:
                raise ValueError(f"recovery_token_id is None for seq_id={seq.seq_id}")
            seq_ids.append(seq.seq_id)
            frontier_versions.append(seq.frontier_version)
            recovery_tokens.append(seq.recovery_token_id)
            temperatures.append(seq.draft_temperature if seq.draft_temperature is not None else seq.temperature)
            target_features.append(seq.extend_dflash_target_features)
            seq.append_token(seq.recovery_token_id)

        entries, diag = self.ddtree_worker.speculate(
            seq_ids=seq_ids,
            frontier_versions=frontier_versions,
            recovery_tokens=recovery_tokens,
            temperatures=temperatures,
            target_features=target_features,
        )
        for seq, entry in zip(seqs, entries):
            entry.verify_input_ids, entry.verify_positions = compile_verify_inputs(
                recovery_token=entry.recovery_token,
                node_token_ids=entry.node_token_ids,
                node_depths=entry.node_depths,
                num_nodes=entry.num_nodes,
                prefix_len=seq.num_cached_tokens,
                device=self.device,
            )
        return SpeculateResult(
            speculations=torch.empty(0, dtype=torch.int64, device=self.device),
            logits_q=torch.empty(0, device=self.device),
            ddtree_diag=diag,
            ddtree_entries=entries,
        )
