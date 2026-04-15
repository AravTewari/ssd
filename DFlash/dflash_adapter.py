"""
DFlash inference adapter for the SSD engine.

This adapter plugs into the SpeculatorSyncDiffusion pipeline as a drop-in
replacement for the LLaDA / Dream diffusion adapters. It uses a trained
DFlashDraftModel (from DFlash/dflash_model.py) together with the target
model's intermediate hidden states to speculatively draft `lookahead` tokens.

Interface contract (matches LLaDADiffusionAdapter / DreamDiffusionAdapter):
  .speculate(seqs, lookahead) -> (draft_tokens [B, k], draft_logits [B, k, V])

How it works:
  1. We receive a batch of Sequence objects each holding their current token_ids.
  2. We run the target HuggingFace model on the full context to get intermediate
     hidden states at `target_layer_ids`.
  3. We embed a block of `lookahead` MASK tokens.
  4. We run the DFlash draft model (non-causal over the block) to get logits.
  5. We argmax (temperature=0) to produce draft tokens, returned for verification.

Note: we load a separate HuggingFace copy of the target specifically for hidden
state extraction (frozen, bf16). This mirrors how Dream/LLaDA adapters load their
own model copy, and avoids depending on the SSD engine's internal model format.
On B200 the 32B model fits comfortably in a single GPU's HBM3e (192 GB).
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ssd.config import Config
from ssd.engine.sequence import Sequence

# Import the draft model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from DFlash.dflash_model import DFlashDraftModel, extract_target_features


def _load_draft_model(checkpoint_dir: str, target_config) -> DFlashDraftModel:
    """Load a trained DFlash draft model from a checkpoint directory."""
    ckpt = Path(checkpoint_dir)
    cfg_path = ckpt / "dflash_config.json"

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"dflash_config.json not found in {checkpoint_dir}. "
            "Make sure you point --draft to a DFlash checkpoint directory."
        )

    with open(cfg_path) as f:
        dflash_cfg = json.load(f)

    draft = DFlashDraftModel(
        target_config=target_config,
        num_draft_layers=dflash_cfg["num_draft_layers"],
        block_size=dflash_cfg["block_size"],
        mask_token_id=dflash_cfg["mask_token_id"],
    )

    weights_path = ckpt / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"pytorch_model.bin not found in {checkpoint_dir}")

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Filter out shared modules (embed_tokens, lm_head) — they come from target
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("embed_tokens") and not k.startswith("lm_head")}
    missing, unexpected = draft.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys loading DFlash draft: {unexpected}")
    non_shared_missing = [k for k in missing if not k.startswith(("embed_tokens", "lm_head"))]
    if non_shared_missing:
        raise RuntimeError(f"Missing keys loading DFlash draft: {non_shared_missing}")

    return draft


class DFlashDraftAdapter:
    """
    DFlash speculative draft adapter.

    Parameters
    ----------
    config : Config
        Engine config. `config.draft` must point to a DFlash checkpoint dir.
        `config.model` is the target model path (used to load HF target for hiddens).
    target_tokenizer : AutoTokenizer
    device : torch.device
    metrics : dict | None
    """

    def __init__(
        self,
        config: Config,
        target_tokenizer: AutoTokenizer,
        device: torch.device,
        metrics: Optional[dict] = None,
        # Unused — kept for signature compatibility with engine wiring
        target_model=None,
    ):
        self.config = config
        self.device = device
        self.metrics = metrics

        target_config = config.hf_config
        dtype = getattr(target_config, "torch_dtype", None) or torch.bfloat16

        # Load a frozen HuggingFace target for hidden-state extraction.
        # With multi-GPU TP the engine already occupies all assigned GPUs, so we
        # spread the HF copy across all available CUDA devices using device_map="auto",
        # leaving headroom for existing KV cache allocations.
        print("[DFlashDraftAdapter] Loading frozen HF target for hidden-state extraction …")
        num_gpus = getattr(config, "num_gpus", 1)
        if num_gpus > 1:
            max_mem = {}
            for i in range(num_gpus):
                free = torch.cuda.mem_get_info(i)[0] // (1024 ** 2)
                # leave 2 GB headroom, use the rest for the HF target
                max_mem[i] = f"{max(0, free - 2048)}MiB"
            self.hf_target = AutoModelForCausalLM.from_pretrained(
                config.model,
                torch_dtype=dtype,
                device_map="auto",
                max_memory=max_mem,
                attn_implementation="sdpa",
            )
        else:
            self.hf_target = AutoModelForCausalLM.from_pretrained(
                config.model,
                torch_dtype=dtype,
                device_map={"": device},
                attn_implementation="sdpa",
            )
        self.hf_target.eval()
        for p in self.hf_target.parameters():
            p.requires_grad_(False)

        # Draft model lives on the primary device passed in (cuda:0 for rank-0 engine)
        draft = _load_draft_model(config.draft, target_config)
        draft = draft.to(device=device, dtype=dtype)
        draft.eval()

        # Share embed_tokens and lm_head from the HF target
        draft.set_shared_modules(
            self.hf_target.model.embed_tokens,
            self.hf_target.lm_head,
        )
        self.draft = draft
        self.tokenizer = target_tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.mask_token_id = draft.mask_token_id
        self.block_size = draft.block_size
        self.vocab_size = target_config.vocab_size

    @torch.inference_mode()
    def speculate(
        self,
        seqs: list[Sequence],
        lookahead: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate `lookahead` draft tokens for each sequence.

        Returns
        -------
        draft_tokens : LongTensor [B, lookahead]
        draft_logits : FloatTensor [B, lookahead, V]
        """
        t0 = perf_counter()
        B = len(seqs)

        # Left-pad sequences to same length
        max_len = max(len(s.token_ids) for s in seqs)
        pad_id = self.tokenizer.pad_token_id
        padded = []
        for s in seqs:
            pad_len = max_len - len(s.token_ids)
            padded.append([pad_id] * pad_len + list(s.token_ids))
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)  # [B, T]

        # Build attention mask (1 for real tokens, 0 for pad)
        attention_mask = (input_ids != pad_id).long()

        # Build the masked block: append `lookahead` MASK tokens to each sequence
        mask_block = torch.full(
            (B, lookahead), self.mask_token_id, dtype=torch.long, device=self.device
        )
        full_ids = torch.cat([input_ids, mask_block], dim=1)  # [B, T + lookahead]
        full_mask = torch.cat(
            [attention_mask, torch.ones(B, lookahead, device=self.device, dtype=torch.long)],
            dim=1,
        )
        T = full_ids.shape[1]
        block_start = input_ids.shape[1]

        # Run HF target to get hidden states (no kv-cache, full context).
        # When hf_target uses device_map="auto" its embed layer is on its first device.
        hf_first_device = next(self.hf_target.parameters()).device
        target_out = self.hf_target(
            input_ids=full_ids.to(hf_first_device),
            attention_mask=full_mask.to(hf_first_device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract tapped hidden states for the CONTEXT (positions 0..block_start-1).
        # The draft model uses ctx features as keys/values and block noise_emb as queries.
        target_features_full = extract_target_features(
            target_out.hidden_states, self.draft.target_layer_ids
        )  # [B, T, n_taps*H]
        ctx_target_feats = target_features_full[:, :block_start, :].to(self.device)  # [B, ctx_len, n*H]

        # Embed the mask block tokens; move to draft device
        noise_emb = self.hf_target.model.embed_tokens(
            full_ids[:, block_start:]
        ).to(self.device)  # [B, lookahead, H]

        # Position ids: [0..ctx_len-1, ctx_len..ctx_len+lookahead-1]
        # draft.forward expects [B, ctx_len + block_len] where first ctx_len are context positions
        position_ids = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)  # [B, ctx+block]

        # Draft forward
        draft_hidden = self.draft(
            noise_embedding=noise_emb,
            target_features=ctx_target_feats,
            position_ids=position_ids,
        )  # [B, lookahead, H]

        # lm_head may be on a different device (shared from hf_target); move input to it
        lm_head_device = next(self.hf_target.lm_head.parameters()).device
        draft_logits = self.hf_target.lm_head(draft_hidden.to(lm_head_device))  # [B, lookahead, V]
        draft_logits = draft_logits.to(self.device)
        draft_tokens = draft_logits.argmax(dim=-1)  # [B, lookahead]

        if self.metrics is not None:
            self.metrics["dflash_draft_step_times"].append(perf_counter() - t0)

        return draft_tokens, draft_logits
