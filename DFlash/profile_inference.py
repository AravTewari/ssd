"""
DFlash inference profiler.

Measures tokens/sec and acceptance rate for the DFlash draft model
against the Qwen3-32B target, bypassing the SSD engine entirely.
Uses the spec_generate loop from the DFlash paper directly.

Usage:
  python DFlash/profile_inference.py \
    --target /path/to/Qwen3-32B \
    --draft  /path/to/dflash_qwen3_32b/step_2500 \
    --device cuda:4 \
    --prompts 20 \
    --max_new_tokens 256

Baselines also measured:
  - Greedy AR (target only, no speculation)
  - DFlash speculative (draft + target verify)

Reported metrics:
  - Tokens/sec (end-to-end wall time)
  - Mean accepted tokens per step (DFlash only)
  - Acceptance rate (fraction of draft tokens accepted)
  - Speedup vs AR baseline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from DFlash.dflash_model import DFlashDraftModel, extract_target_features, build_target_layer_ids

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    "Explain the concept of speculative decoding in language models.",
    "Write a Python function to compute the nth Fibonacci number efficiently.",
    "What are the main differences between transformers and RNNs?",
    "Describe the steps to train a large language model from scratch.",
    "Summarize the key ideas behind the attention mechanism in neural networks.",
    "How does RLHF improve language model alignment with human preferences?",
    "Write a SQL query to find the top 5 customers by total purchase amount.",
    "Explain gradient descent and its variants (SGD, Adam, AdaGrad).",
    "What is the difference between supervised and unsupervised learning?",
    "Describe how mixture-of-experts models work and their advantages.",
    "Write a regex pattern that matches valid email addresses.",
    "How does flash attention reduce memory usage compared to standard attention?",
    "Explain the key differences between BERT and GPT architectures.",
    "What is quantization in deep learning and when should you use it?",
    "Describe the architecture of a modern transformer-based language model.",
    "How does KV caching speed up autoregressive generation?",
    "Write a Python class implementing a doubly linked list.",
    "Explain the concept of tensor parallelism in distributed training.",
    "What are the trade-offs between model size and inference latency?",
    "Describe how chain-of-thought prompting improves reasoning in LLMs.",
]


# ---------------------------------------------------------------------------
# Greedy AR baseline
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_ar(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
) -> tuple[torch.Tensor, float]:
    """Standard greedy autoregressive generation. Returns (output_ids, tokens_per_sec)."""
    device = input_ids.device
    generated = input_ids.clone()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        out = model(input_ids=generated, use_cache=False, return_dict=True)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
        if next_tok.item() == eos_token_id:
            break
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    n_new = generated.shape[1] - input_ids.shape[1]
    return generated, n_new / elapsed


# ---------------------------------------------------------------------------
# DFlash speculative generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_dflash(
    target,
    draft: DFlashDraftModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    block_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, float, list[int]]:
    """
    DFlash speculative generation.
    Returns (output_ids, tokens_per_sec, acceptance_lengths).
    """
    device = input_ids.device
    B, prompt_len = input_ids.shape
    max_len = prompt_len + max_new_tokens

    # Prefill target
    prefill_out = target(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    # Greedy sample first token
    first_token = prefill_out.logits[:, -1:, :].argmax(dim=-1)  # [B, 1]
    generated = torch.cat([input_ids, first_token], dim=1)      # [B, prompt_len+1]

    acceptance_lengths = []

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    while generated.shape[1] < max_len:
        cur_len = generated.shape[1]
        remaining = max_len - cur_len
        bs = min(block_size, remaining)

        # Build masked block appended to current sequence
        mask_block = torch.full((B, bs), mask_token_id, dtype=torch.long, device=device)
        full_ids = torch.cat([generated, mask_block], dim=1)   # [B, cur_len + bs]
        block_start = cur_len

        # Target forward for hidden states
        target_out = target(
            input_ids=full_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract context features (prefix before block)
        target_features_full = extract_target_features(
            target_out.hidden_states, draft.target_layer_ids
        )
        ctx_feats = target_features_full[:, :block_start, :]   # [B, cur_len, n*H]

        # Noise embedding for the mask block
        noise_emb = target.model.embed_tokens(full_ids[:, block_start:])  # [B, bs, H]

        # Position ids: [0..cur_len+bs-1]
        draft_dev = next(draft.parameters()).device
        pos_ids = torch.arange(cur_len + bs, device=draft_dev).unsqueeze(0).expand(B, -1)

        # Draft forward — move inputs to draft device in case target used a different GPU
        draft_hidden = draft(
            noise_embedding=noise_emb.to(draft_dev),
            target_features=ctx_feats.to(draft_dev),
            position_ids=pos_ids,
        )
        draft_logits = target.lm_head(draft_hidden.to(next(target.lm_head.parameters()).device))  # [B, bs, V]
        draft_tokens = draft_logits.argmax(dim=-1)              # [B, bs]

        # Build speculative sequence: current + draft tokens
        spec_ids = torch.cat([generated, draft_tokens], dim=1) # [B, cur_len+bs]

        # Verify with target (single forward on speculative sequence)
        verify_out = target(
            input_ids=spec_ids,
            use_cache=False,
            return_dict=True,
        )
        target_tokens = verify_out.logits[:, cur_len-1:-1, :].argmax(dim=-1)  # [B, bs]

        # Cascaded acceptance: accept prefix up to first mismatch
        match = (draft_tokens == target_tokens)                 # [B, bs]
        # Find first mismatch per batch item (batch size 1 here)
        match_row = match[0]
        if match_row.all():
            n_accept = bs
        else:
            n_accept = match_row.long().cumprod(0).sum().item()

        # Append accepted tokens + one correction token
        accepted = draft_tokens[:, :n_accept]
        correction = target_tokens[:, n_accept:n_accept+1]
        generated = torch.cat([generated, accepted, correction], dim=1)

        acceptance_lengths.append(n_accept)

        if eos_token_id in generated[0, prompt_len:]:
            break

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    n_new = generated.shape[1] - prompt_len - 1  # subtract prefill first token
    tps = n_new / elapsed if elapsed > 0 else 0.0
    return generated[:, :max_len], tps, acceptance_lengths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--draft",  required=True, help="DFlash checkpoint directory")
    p.add_argument("--device", default="cuda:4")
    p.add_argument("--prompts", type=int, default=10, help="Number of prompts to run")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    # With device_map="auto", target layers start on cuda:0.
    # Draft must be on the same device as the target's first layer (cuda:0)
    # so that hidden states / features don't cross devices unexpectedly.
    draft_device = torch.device("cuda:0")

    print(f"Device : {device}")
    print(f"Draft device: {draft_device}")
    print(f"Target : {args.target}")
    print(f"Draft  : {args.draft}")
    print(f"Prompts: {args.prompts}  max_new_tokens: {args.max_new_tokens}")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    # Load target
    print("Loading target model...")
    # Use device_map="auto" to distribute across all visible GPUs.
    # Forcing device_map={"": device} requires the full model (~64GB for 32B bf16)
    # to fit on a single GPU, which causes hangs on insufficient VRAM.
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=dtype, device_map="auto"
    )
    print("BEFORE Target eval")
    target.eval()
    print("  Target loaded.")

    # Load DFlash draft
    import json
    cfg_path = Path(args.draft) / "dflash_config.json"
    with open(cfg_path) as f:
        dcfg = json.load(f)

    print("  Building draft model...")
    from transformers import AutoConfig
    target_config = target.config
    draft = DFlashDraftModel(
        target_config=target_config,
        num_draft_layers=dcfg["num_draft_layers"],
        block_size=dcfg["block_size"],
        mask_token_id=dcfg["mask_token_id"],
    )
    print("  Loading draft checkpoint...")
    sd = torch.load(Path(args.draft) / "pytorch_model.bin", map_location="cpu", weights_only=False)
    sd = {k: v for k, v in sd.items() if not k.startswith(("embed_tokens", "lm_head"))}
    draft.load_state_dict(sd, strict=False)
    draft = draft.to(device=draft_device, dtype=dtype)
    draft.eval()
    draft.set_shared_modules(target.model.embed_tokens, target.lm_head)
    print(f"  Draft: {dcfg['num_draft_layers']} layer(s), block_size={dcfg['block_size']}, step={dcfg['training_step']}")
    print()

    prompts = PROMPTS[:args.prompts]

    # -----------------------------------------------------------------------
    # AR baseline
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("BASELINE: Greedy AR (target only)")
    print("=" * 60)
    ar_tps_list = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        # Warmup on first prompt
        if i == 0:
            _ = generate_ar(target, input_ids, 8, eos_id)

        _, tps = generate_ar(target, input_ids, args.max_new_tokens, eos_id)
        ar_tps_list.append(tps)
        print(f"  [{i+1:2d}/{args.prompts}] {tps:.1f} tok/s")

    ar_mean = mean(ar_tps_list)
    ar_std  = stdev(ar_tps_list) if len(ar_tps_list) > 1 else 0.0
    print(f"\n  AR mean: {ar_mean:.1f} ± {ar_std:.1f} tok/s\n")

    # -----------------------------------------------------------------------
    # DFlash speculative
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(f"DFLASH: block_size={dcfg['block_size']}, step={dcfg['training_step']}")
    print("=" * 60)
    dflash_tps_list = []
    all_accept_lens = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        # Warmup
        if i == 0:
            _ = generate_dflash(target, draft, input_ids, 8, eos_id,
                                 dcfg["block_size"], dcfg["mask_token_id"])

        _, tps, accept_lens = generate_dflash(
            target, draft, input_ids, args.max_new_tokens, eos_id,
            dcfg["block_size"], dcfg["mask_token_id"],
        )
        dflash_tps_list.append(tps)
        all_accept_lens.extend(accept_lens)
        avg_al = mean(accept_lens) if accept_lens else 0.0
        print(f"  [{i+1:2d}/{args.prompts}] {tps:.1f} tok/s  avg_accept={avg_al:.2f}/{dcfg['block_size']}")

    dflash_mean = mean(dflash_tps_list)
    dflash_std  = stdev(dflash_tps_list) if len(dflash_tps_list) > 1 else 0.0
    overall_accept = mean(all_accept_lens) if all_accept_lens else 0.0
    accept_rate = overall_accept / dcfg["block_size"]

    print(f"\n  DFlash mean: {dflash_mean:.1f} ± {dflash_std:.1f} tok/s")
    print(f"  Mean accepted tokens/step: {overall_accept:.2f} / {dcfg['block_size']}")
    print(f"  Acceptance rate: {accept_rate:.1%}")
    print(f"  Speedup vs AR: {dflash_mean/ar_mean:.2f}x")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  AR baseline      : {ar_mean:.1f} tok/s")
    print(f"  DFlash (step {dcfg['training_step']:5d}): {dflash_mean:.1f} tok/s")
    print(f"  Speedup          : {dflash_mean/ar_mean:.2f}x")
    print(f"  Acceptance rate  : {accept_rate:.1%}  ({overall_accept:.2f}/{dcfg['block_size']} tokens/step)")
    print(f"  Block size       : {dcfg['block_size']}")
    print(f"  Draft layers     : {dcfg['num_draft_layers']}")
    print(f"  Training step    : {dcfg['training_step']} / 5000")
    print(f"  Note: model is ~60% trained — acceptance rate will improve after full run")


if __name__ == "__main__":
    main()
