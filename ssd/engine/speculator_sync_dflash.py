import torch

from ssd.engine.dflash_worker import DFlashWorkerHandle
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase
from ssd.engine.sequence import Sequence


class SpeculatorSyncDFlash(SpeculatorBase):
    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        dflash_worker: DFlashWorkerHandle,
    ):
        super().__init__(lookahead, device)
        self.dflash_worker = dflash_worker

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        if verify_result.dflash_target_features is None:
            raise RuntimeError("DFlash prefill requires prompt target features from the verifier")
        for seq, prompt_features in zip(seqs, verify_result.dflash_target_features):
            seq.frontier_version = 0
            seq.last_dflash_target_feature = prompt_features[-1].clone()
            seq.extend_dflash_target_features = None
            seq.extend_dflash_token_ids = None
            seq.extend_dflash_count = 0
        self.dflash_worker.prefill(
            seq_ids=[seq.seq_id for seq in seqs],
            prompt_target_features=verify_result.dflash_target_features,
        )
        return SpeculateResult([], [])

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        batch_size = len(seqs)
        speculations = torch.zeros(
            batch_size,
            self.lookahead + 1,
            dtype=torch.int64,
            device=self.device,
        )

        seq_ids = []
        frontier_versions = []
        recovery_tokens = []
        temperatures = []
        target_features = []
        for i, seq in enumerate(seqs):
            if seq.recovery_token_id is None:
                raise ValueError(f"recovery_token_id is None for seq {i}")
            seq_ids.append(seq.seq_id)
            frontier_versions.append(seq.frontier_version)
            recovery_tokens.append(seq.recovery_token_id)
            temperatures.append(seq.draft_temperature if seq.draft_temperature is not None else seq.temperature)
            seq.append_token(seq.recovery_token_id)
            target_features.append(seq.extend_dflash_target_features)
        speculations[:, 0] = torch.tensor(recovery_tokens, dtype=torch.int64, device=self.device)

        draft_tokens, logits_q, block_hidden = self.dflash_worker.speculate(
            seq_ids=seq_ids,
            frontier_versions=frontier_versions,
            recovery_tokens=recovery_tokens,
            temperatures=temperatures,
            target_features=target_features,
        )
        speculations[:, 1:] = draft_tokens

        for i, seq in enumerate(seqs):
            seq.token_ids.extend(draft_tokens[i].tolist())
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
            seq.num_draft_cached_tokens += self.lookahead + 1

        return SpeculateResult(speculations, logits_q, dflash_block_hidden=block_hidden)
