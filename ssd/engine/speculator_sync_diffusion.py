from typing import Any

import torch

from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase
from ssd.engine.sequence import Sequence


class SpeculatorSyncDiffusion(SpeculatorBase):
    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        diffusion_adapter: Any,
    ):
        super().__init__(lookahead, device)
        self.diffusion_adapter = diffusion_adapter

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        return SpeculateResult([], [])

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        batch_size = len(seqs)
        speculations = torch.zeros(
            batch_size,
            self.lookahead + 1,
            dtype=torch.int64,
            device=self.device,
        )

        recovery_tokens = []
        for i, seq in enumerate(seqs):
            if seq.recovery_token_id is None:
                raise ValueError(f"recovery_token_id is None for seq {i}")
            recovery_tokens.append(seq.recovery_token_id)
            seq.append_token(seq.recovery_token_id)
        speculations[:, 0] = torch.tensor(recovery_tokens, dtype=torch.int64, device=self.device)

        draft_tokens, logits_q = self.diffusion_adapter.speculate(seqs, self.lookahead)
        speculations[:, 1:] = draft_tokens

        for i, seq in enumerate(seqs):
            seq.token_ids.extend(draft_tokens[i].tolist())
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
            seq.num_draft_cached_tokens += self.lookahead + 1

        return SpeculateResult(speculations, logits_q)
