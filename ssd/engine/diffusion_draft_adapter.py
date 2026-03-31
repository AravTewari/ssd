from __future__ import annotations

from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

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
        # Try loading in order: MaskedLM (MDLM) > CausalLM (Dream) > Model (fallback)
        for auto_cls in (AutoModelForMaskedLM, AutoModelForCausalLM, AutoModel):
            try:
                self.model = auto_cls.from_pretrained(
                    config.draft,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                ).to(device).eval()
                break
            except (ValueError, KeyError):
                continue
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.draft,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self._validate_compatibility(target_tokenizer)

        # Detect model type to select correct forward/logits convention
        model_type = getattr(config.draft_hf_config, "model_type", "").lower()
        self._is_dream = "dream" in model_type
        # Dream: positional args (x, attn_mask, tok_idx), needs logits shift
        # MDLM/others: keyword args (input_ids=x, attention_mask=mask), no logits shift

    def _validate_compatibility(self, target_tokenizer: AutoTokenizer):
        # The critical check: tokenization must produce the same ids for normal text.
        # Diffusion models (Dream, LLaDA) may add special tokens that inflate
        # vocab_size, so we compare tokenizer.vocab_size (base vocab) rather than
        # config.vocab_size (model embedding dimension).
        probe = "Speculative decoding compatibility probe."
        target_ids = target_tokenizer.encode(probe, add_special_tokens=False)
        draft_ids = self.tokenizer.encode(probe, add_special_tokens=False)
        if target_ids != draft_ids:
            raise ValueError(
                "diffusion draft requires identical tokenization for target and draft tokenizers. "
                f"target encoded to {target_ids[:8]}... draft encoded to {draft_ids[:8]}..."
            )

        mask_id = self.config.diffusion_mask_id
        model_vocab = self.config.draft_hf_config.vocab_size
        if mask_id < 0 or mask_id >= model_vocab:
            raise ValueError(
                f"diffusion_mask_id={mask_id} is out of range for model vocab size {model_vocab}"
            )

        if target_tokenizer.pad_token_id == mask_id:
            raise ValueError(
                "diffusion_mask_id must not collide with target pad_token_id"
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
        attention_mask = batch["attention_mask"].to(self.device).bool()

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
        """Masked-diffusion denoising loop following Dream's generation logic.

        Returns (draft_tokens [B, gen_length], draft_logits [B, gen_length, V]).
        """
        if gen_length <= 0:
            raise ValueError("gen_length must be > 0")
        if steps <= 0:
            raise ValueError("diffusion steps must be > 0")

        batch_size, prompt_len = prompt.shape
        total_len = prompt_len + gen_length

        # Build input: [prompt tokens | MASK MASK ... MASK]
        x = torch.full(
            (batch_size, total_len), mask_id, dtype=torch.long, device=self.device,
        )
        x[:, :prompt_len] = prompt

        # Prepare attention mask.  Dream expects either the string "full" (no
        # padding) or a 4D bool mask [B, 1, N, N] when there IS padding.
        full_attn_2d = torch.cat(
            [
                attention_mask,
                torch.ones(batch_size, gen_length, dtype=torch.bool, device=self.device),
            ],
            dim=-1,
        )
        has_padding = not full_attn_2d.all()
        if has_padding:
            # 4D bidirectional bool mask [B, 1, N, N]
            attn_mask = torch.logical_and(
                full_attn_2d.unsqueeze(1).unsqueeze(-2),
                full_attn_2d.unsqueeze(1).unsqueeze(-1),
            )
            # tok_idx: position ids that skip padding positions
            tok_idx = full_attn_2d.long().cumsum(-1) - 1
            tok_idx.masked_fill_(~full_attn_2d, 1)
        else:
            attn_mask = "full"
            tok_idx = None

        # Continuous timestep schedule (Dream convention)
        eps = 1e-3
        timesteps = torch.linspace(1, eps, steps + 1, device=self.device)

        final_logits = None
        for i in range(steps):
            mask_index = x == mask_id

            if self._is_dream:
                logits = self.model(x, attn_mask, tok_idx).logits
                # Dream predicts next-token; shift logits right by 1
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            else:
                # MDLM / standard HF: keyword args, predicts at-position (no shift)
                logits = self.model(input_ids=x, attention_mask=full_attn_2d.long()).logits
            final_logits = logits

            t = timesteps[i]
            s = timesteps[i + 1]

            if remasking == "low_confidence":
                # MaskGIT-style: reveal highest-confidence tokens each step.
                confidence, x0 = F.softmax(logits.float(), dim=-1).max(dim=-1)
                num_mask = mask_index.sum()
                n_transfer = int(num_mask / batch_size * (1 - s / t)) if i < steps - 1 else int(num_mask / batch_size)
                full_conf = torch.full(x.shape, float("-inf"), dtype=torch.float32, device=self.device)
                full_conf[mask_index] = confidence[mask_index]
                if n_transfer > 0:
                    _, transfer_idx = torch.topk(full_conf.view(batch_size, -1), n_transfer, dim=-1)
                    row_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand_as(transfer_idx)
                    x_new = torch.full_like(x, mask_id)
                    x_new[mask_index] = x0[mask_index]
                    x.view(batch_size, -1)[row_idx, transfer_idx] = x_new.view(batch_size, -1)[row_idx, transfer_idx]
            else:
                # Origin schedule: probabilistic transfer per masked token.
                p_transfer = (1 - s / t) if i < steps - 1 else 1.0
                x0 = torch.full_like(x, mask_id)
                _, sampled = F.softmax(logits.float(), dim=-1).max(dim=-1)
                transfer = torch.rand_like(x, dtype=torch.float) < p_transfer
                x0[mask_index & transfer] = sampled[mask_index & transfer]
                x[mask_index] = x0[mask_index]

        assert final_logits is not None
        return x[:, -gen_length:], final_logits[:, -gen_length:, :]
