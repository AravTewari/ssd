from __future__ import annotations

from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ssd.config import Config
from ssd.engine.sequence import Sequence


class LLaDADiffusionAdapter:
    def __init__(
        self,
        config: Config,
        target_tokenizer: AutoTokenizer,
        device: torch.device,
        metrics: dict | None = None,
    ):
        self.config = config
        self.device = device
        self.metrics = metrics

        dtype = getattr(config.draft_hf_config, "torch_dtype", None) or torch.bfloat16
        self.model = AutoModel.from_pretrained(
            config.draft,
            trust_remote_code=True,
            dtype=dtype,
        ).to(device).eval()
        self.model_vocab_size = int(self.model.config.vocab_size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.draft,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self._validate_compatibility(target_tokenizer)

    def _validate_compatibility(self, target_tokenizer: AutoTokenizer):
        target_vocab = len(target_tokenizer)
        if target_vocab > self.model_vocab_size:
            raise ValueError(
                "llada_diffusion requires the target tokenizer ids to fit within the draft model vocab, "
                f"got target_size={target_vocab} draft_model_vocab={self.model_vocab_size}"
            )

        if (
            target_tokenizer.eos_token_id is not None
            and self.tokenizer.eos_token_id is not None
            and target_tokenizer.eos_token_id != self.tokenizer.eos_token_id
        ):
            raise ValueError(
                "llada_diffusion requires matching EOS token ids between target and draft"
            )

        if (
            target_tokenizer.pad_token_id is not None
            and self.tokenizer.pad_token_id is not None
            and target_tokenizer.pad_token_id != self.tokenizer.pad_token_id
        ):
            raise ValueError(
                "llada_diffusion requires matching PAD token ids between target and draft"
            )

        probe = "Speculative decoding compatibility probe."
        target_ids = target_tokenizer.encode(probe, add_special_tokens=False)
        draft_ids = self.tokenizer.encode(probe, add_special_tokens=False)
        if target_ids != draft_ids:
            raise ValueError(
                "llada_diffusion requires identical tokenization for target and draft tokenizers"
            )

        mask_id = self.config.diffusion_mask_id
        if mask_id < 0 or mask_id >= self.model_vocab_size:
            raise ValueError(
                f"Configured diffusion_mask_id={mask_id} is out of range for model vocab size {self.model_vocab_size}"
            )

        if target_tokenizer.pad_token_id == mask_id:
            raise ValueError(
                "Configured diffusion_mask_id must not match the target tokenizer pad token id"
            )

    @torch.inference_mode()
    def speculate(
        self,
        seqs: list[Sequence],
        lookahead: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenizer.pad(
            {"input_ids": [seq.token_ids for seq in seqs]},
            padding=True,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        t0 = perf_counter()
        draft_tokens, draft_logits = self._generate_with_logits(
            input_ids,
            attention_mask,
            gen_length=lookahead,
            steps=self.config.diffusion_steps,
            mask_id=self.config.diffusion_mask_id,
            remasking=self.config.diffusion_remasking,
        )
        if self.metrics is not None:
            self.metrics["diffusion_draft_step_times"].append(perf_counter() - t0)
        return draft_tokens, draft_logits

    def _generate_with_logits(
        self,
        prompt: torch.Tensor,
        attention_mask: torch.Tensor,
        gen_length: int,
        steps: int,
        mask_id: int,
        remasking: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if remasking != "low_confidence":
            raise ValueError(f"Unsupported remasking mode: {remasking}")
        if gen_length <= 0:
            raise ValueError("gen_length must be > 0")
        if steps <= 0:
            raise ValueError("diffusion steps must be > 0")

        batch_size, prompt_len = prompt.shape
        total_len = prompt_len + gen_length
        x = torch.full(
            (batch_size, total_len),
            mask_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, :prompt_len] = prompt

        full_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, gen_length),
                    dtype=attention_mask.dtype,
                    device=self.device,
                ),
            ],
            dim=-1,
        )

        num_transfer_tokens = self._get_num_transfer_tokens(
            x[:, prompt_len:] == mask_id,
            steps,
        )
        final_logits = None
        for step_idx in range(steps):
            mask_index = x == mask_id
            logits = self.model(x, attention_mask=full_attention_mask).logits
            final_logits = logits
            x0 = torch.argmax(logits, dim=-1)

            probs = F.softmax(logits.float(), dim=-1)
            x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, total_len:] = float("-inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, float("-inf")))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=self.device)
            for row in range(batch_size):
                k = int(num_transfer_tokens[row, step_idx].item())
                if k <= 0:
                    continue
                _, select_index = torch.topk(confidence[row], k=k)
                transfer_index[row, select_index] = True
            x[transfer_index] = x0[transfer_index]

        assert final_logits is not None
        return x[:, -gen_length:], final_logits[:, -gen_length:, :]

    @staticmethod
    def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = torch.zeros(
            mask_num.size(0),
            steps,
            device=mask_index.device,
            dtype=torch.int64,
        ) + base
        for row in range(mask_num.size(0)):
            rem = int(remainder[row].item())
            if rem > 0:
                num_transfer_tokens[row, :rem] += 1
        return num_transfer_tokens
