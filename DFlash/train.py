"""
DFlash training script for Qwen3-32B.

Usage (single B200 node, 8 GPUs via torchrun):
  torchrun --nproc_per_node=8 DFlash/train.py \
    --target  /path/to/Qwen3-32B \
    --out_dir /path/to/dflash_qwen3_32b \
    --dataset alpaca \
    --block_size 4 \
    --num_draft_layers 1 \
    --batch_size 2 \
    --grad_accum 4 \
    --max_steps 5000 \
    --lr 1e-4 \
    --save_every 500

Training objective (DFlash / BlockDiffusion):
  For each training sequence of length T, we randomly sample a contiguous block
  of `block_size` tokens and mask them with the MASK token id.  The draft model
  receives:
    - the target model's intermediate hidden states for the full (unmasked) sequence,
    - the token embeddings of the masked block (MASK tokens),
  and must predict all `block_size` original tokens in parallel.

  Loss = cross-entropy over the block tokens (averaged over block positions).

Key design choices matching the DFlash paper:
  - Draft layers reuse the target's config (hidden size, num_heads, head_dim,
    rope params), so no projection mismatch.
  - Target model is kept frozen (eval mode, no grad) — we only train the draft.
  - Intermediate target hidden states are tapped at evenly-spaced layers
    (`target_layer_ids`) and concatenated before the draft `fc` projection.
  - We use FSDP2 (torch.distributed.fsdp) to shard the frozen target across GPUs
    and train the draft in full precision (bf16).
  - On B200 (SM100), flash-attention via F.scaled_dot_product_attention uses
    the hardware FlashAttention-3 path automatically with torch ≥ 2.4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# Make sure the repo root is importable when running from DFlash/
sys.path.insert(0, str(Path(__file__).parent.parent))

from DFlash.dflash_model import DFlashDraftModel, extract_target_features


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

JSONL_DATASETS = {
    "alpaca":        "alpaca/alpaca_data_10000.jsonl",
    "humaneval":     "humaneval/humaneval_data_10000.jsonl",
    "gsm":           "gsm8k/gsm8k_data_10000.jsonl",
    "ultrafeedback": "ultrafeedback/ultrafeedback_data_10000.jsonl",
    "c4":            "c4/c4_data_10000.jsonl",
}


class TokenBlockDataset(IterableDataset):
    """
    Streams tokenized sequences from one or more .jsonl files.
    Each line is expected to have a "text" or "content" or "prompt"+"response" field.
    Sequences are chunked to `max_seq_len` tokens with no padding.
    """

    def __init__(
        self,
        paths: list[str],
        tokenizer,
        max_seq_len: int = 4096,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.world_size = world_size

    def _text_from_row(self, row: dict) -> str:
        for key in ("text", "content", "response", "prompt", "question", "output"):
            if key in row and isinstance(row[key], str) and row[key].strip():
                return row[key]
        # fallback: dump the whole dict
        return " ".join(str(v) for v in row.values() if isinstance(v, str))

    def __iter__(self):
        buf: list[int] = []
        for path_idx, path in enumerate(self.paths):
            with open(path) as f:
                for line_idx, line in enumerate(f):
                    # shard across ranks by line index
                    if (path_idx * 100000 + line_idx) % self.world_size != self.rank:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = self._text_from_row(row)
                    ids = self.tokenizer.encode(text, add_special_tokens=True)
                    buf.extend(ids)
                    while len(buf) >= self.max_seq_len:
                        yield torch.tensor(buf[: self.max_seq_len], dtype=torch.long)
                        buf = buf[self.max_seq_len :]


def build_dataloader(
    dataset_name: str,
    dataset_dir: str,
    tokenizer,
    max_seq_len: int,
    batch_size: int,
    rank: int,
    world_size: int,
) -> DataLoader:
    if dataset_name in JSONL_DATASETS:
        rel = JSONL_DATASETS[dataset_name]
        paths = [os.path.join(dataset_dir, rel)]
    else:
        # treat as a direct path to a directory of .jsonl files
        p = Path(dataset_name)
        if p.is_file():
            paths = [str(p)]
        elif p.is_dir():
            paths = sorted(str(f) for f in p.glob("**/*.jsonl"))
        else:
            raise ValueError(f"Unknown dataset '{dataset_name}' and not a path.")

    ds = TokenBlockDataset(paths, tokenizer, max_seq_len=max_seq_len, rank=rank, world_size=world_size)
    return DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def build_mask_schedule(block_size: int, seq_len: int) -> tuple[int, int]:
    """
    Sample a random block start position within the sequence (not at position 0
    so the draft always has some context).
    Returns (block_start, block_end) indices into the sequence.
    """
    lo = max(1, seq_len // 4)          # ensure at least 25% context
    hi = seq_len - block_size - 1      # ensure at least 1 token after block
    if lo >= hi:
        lo = 1
        hi = max(2, seq_len - block_size - 1)
    block_start = random.randint(lo, max(lo, hi))
    return block_start, block_start + block_size


@torch.no_grad()
def run_target(
    target: nn.Module,
    input_ids: torch.Tensor,
    target_layer_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the frozen target model and extract:
      - target_features: [B, S, n_taps*H] concatenated intermediate hiddens
      - target_logits:   [B, S, V]  (used for soft-label distillation)
    """
    out = target(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    features = extract_target_features(out.hidden_states, target_layer_ids)  # [B, S, n*H]
    return features, out.logits


def compute_loss(
    draft_logits: torch.Tensor,       # [B, block_size, V]
    target_ids: torch.Tensor,         # [B, block_size]  ground truth tokens
    target_logits: torch.Tensor,      # [B, block_size, V]  teacher logits (for distillation)
    alpha_ce: float = 1.0,
    alpha_kl: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Combined cross-entropy (hard labels) + reverse-KL distillation (soft labels).

    Following DFlash paper: CE loss drives the draft to predict the correct token,
    KL distillation matches the full output distribution of the target.
    """
    B, block_size, V = draft_logits.shape

    # Hard-label CE
    ce_loss = F.cross_entropy(
        draft_logits.view(-1, V),
        target_ids.view(-1),
        reduction="mean",
    )

    # Soft-label KL (reverse KL: draft || teacher)
    with torch.no_grad():
        teacher_log_probs = F.log_softmax(target_logits.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(draft_logits.float() / temperature, dim=-1)
    # KL(teacher || student) = sum teacher * (log teacher - log student)
    kl_loss = F.kl_div(
        student_log_probs.view(-1, V),
        teacher_log_probs.view(-1, V).exp(),
        reduction="batchmean",
        log_target=False,
    ) * (temperature ** 2)

    return alpha_ce * ce_loss + alpha_kl * kl_loss


# ---------------------------------------------------------------------------
# FSDP / DDP setup
# ---------------------------------------------------------------------------

def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def teardown_distributed():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a DFlash draft model for Qwen3-32B")
    p.add_argument("--target", required=True, help="Path to Qwen3-32B model directory")
    p.add_argument("--out_dir", required=True, help="Where to save draft model checkpoints")
    p.add_argument("--resume", default=None, help="Path to checkpoint directory to resume from (e.g. out_dir/step_2500)")
    p.add_argument("--dataset", default="alpaca",
                   help="Dataset name (alpaca|gsm|humaneval|ultrafeedback|c4) or path to .jsonl dir")
    p.add_argument("--dataset_dir", default=os.environ.get("SSD_DATASET_DIR", ""),
                   help="Root of processed datasets (uses $SSD_DATASET_DIR by default)")

    # Model architecture
    p.add_argument("--num_draft_layers", type=int, default=1,
                   help="Number of DFlash draft transformer layers")
    p.add_argument("--block_size", type=int, default=4,
                   help="Speculative block size (tokens predicted in parallel)")
    p.add_argument("--mask_token_id", type=int, default=151666,
                   help="Mask token id (Qwen3 default: 151666)")

    # Training
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Per-GPU batch size")
    p.add_argument("--grad_accum", type=int, default=8,
                   help="Gradient accumulation steps")
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--clip_grad", type=float, default=1.0)

    # Loss weights
    p.add_argument("--alpha_ce", type=float, default=1.0, help="CE loss weight")
    p.add_argument("--alpha_kl", type=float, default=1.0, help="KL distillation weight")
    p.add_argument("--kl_temperature", type=float, default=1.0)

    # Logging / saving
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb_project", default="dflash-qwen3")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def lr_schedule(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_draft(draft: nn.Module, out_dir: str, step: int, target_config, args, prev_step: int | None = None):
    """Save draft model weights + config, deleting the previous checkpoint to save disk."""
    out = Path(out_dir) / f"step_{step}"
    out.mkdir(parents=True, exist_ok=True)

    # Save weights (unwrap DDP if needed)
    model_to_save = draft.module if hasattr(draft, "module") else draft
    torch.save(model_to_save.state_dict(), out / "pytorch_model.bin")

    # Remove previous checkpoint once new one is fully written
    if prev_step is not None:
        prev = Path(out_dir) / f"step_{prev_step}"
        if prev.exists() and prev != out:
            import shutil
            shutil.rmtree(prev)
            print(f"[save] removed previous checkpoint step_{prev_step}")

    # Save a config that captures the key DFlash hyperparameters
    cfg = {
        "model_type": "dflash_draft",
        "base_model": "Qwen3-32B",
        "num_draft_layers": args.num_draft_layers,
        "block_size": args.block_size,
        "mask_token_id": args.mask_token_id,
        "target_layer_ids": model_to_save.target_layer_ids,
        "hidden_size": target_config.hidden_size,
        "num_attention_heads": target_config.num_attention_heads,
        "num_key_value_heads": target_config.num_key_value_heads,
        "intermediate_size": target_config.intermediate_size,
        "rms_norm_eps": target_config.rms_norm_eps,
        "rope_theta": getattr(target_config, "rope_theta", 1_000_000.0),
        "max_position_embeddings": getattr(target_config, "max_position_embeddings", 131072),
        "vocab_size": target_config.vocab_size,
        "training_step": step,
    }
    with open(out / "dflash_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[save] checkpoint at step {step} → {out}")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Distributed init ---
    rank, world_size = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16  # B200 supports bf16 natively at full throughput

    if is_main:
        print(f"[init] world_size={world_size}  device={device}  dtype={dtype}")
        print(f"[init] target={args.target}")
        print(f"[init] block_size={args.block_size}  num_draft_layers={args.num_draft_layers}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.target, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Target model (frozen, bf16) ---
    if is_main:
        print("[init] Loading frozen target model …")
    target_config = AutoConfig.from_pretrained(args.target)

    # Load target in bf16 and shard across GPUs with device_map="auto" or manual.
    # For a 32B model on 8x B200 (each 192 GB HBM3e) this fits comfortably.
    target = AutoModelForCausalLM.from_pretrained(
        args.target,
        dtype=dtype,
        device_map={"": device},   # place each rank's shard on its own GPU
    )
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)

    if is_main:
        n_target = sum(p.numel() for p in target.parameters()) / 1e9
        print(f"[init] Target params: {n_target:.1f}B (frozen)")

    # --- Draft model (trainable, bf16) ---
    draft = DFlashDraftModel(
        target_config=target_config,
        num_draft_layers=args.num_draft_layers,
        block_size=args.block_size,
        mask_token_id=args.mask_token_id,
    ).to(device=device, dtype=dtype)

    # Share embed_tokens and lm_head with target for parameter-efficiency
    # (these are frozen — the draft only trains its new layers + fc + norms)
    draft.embed_tokens = target.model.embed_tokens
    draft.lm_head = target.lm_head
    draft.set_shared_modules(target.model.embed_tokens, target.lm_head)

    n_draft = sum(p.numel() for p in draft.parameters() if p.requires_grad) / 1e6
    if is_main:
        print(f"[init] Draft trainable params: {n_draft:.1f}M")

    # Wrap draft in DDP
    draft = torch.nn.parallel.DistributedDataParallel(
        draft, device_ids=[rank], find_unused_parameters=False
    )
    draft_inner = draft.module

    # --- Optimizer ---
    # Only optimise the non-shared parameters (layers, fc, norms)
    trainable_params = [p for p in draft.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay, fused=True
    )

    # --- Dataset ---
    loader = build_dataloader(
        dataset_name=args.dataset,
        dataset_dir=args.dataset_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size,
    )
    data_iter = iter(loader)

    # --- Resume from checkpoint ---
    start_step = 0
    last_saved_step = None
    if args.resume is not None:
        resume_path = Path(args.resume)
        ckpt_file = resume_path / "pytorch_model.bin"
        cfg_file = resume_path / "dflash_config.json"
        if ckpt_file.exists():
            if is_main:
                print(f"[resume] Loading checkpoint from {resume_path}")
            sd = torch.load(ckpt_file, map_location=device, weights_only=False)
            sd = {k: v for k, v in sd.items() if not k.startswith(("embed_tokens", "lm_head"))}
            draft_inner.load_state_dict(sd, strict=False)
            if cfg_file.exists():
                with open(cfg_file) as f:
                    resume_cfg = json.load(f)
                start_step = resume_cfg.get("training_step", 0)
                last_saved_step = start_step
            if is_main:
                print(f"[resume] Resuming from step {start_step}")
        else:
            if is_main:
                print(f"[resume] WARNING: no checkpoint found at {ckpt_file}, starting fresh")

    # --- W&B ---
    if args.wandb and is_main:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args), resume="allow")

    # --- Training loop ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer.zero_grad()
    step = start_step
    accum_loss = 0.0
    accum_ce = 0.0
    accum_kl = 0.0

    if is_main:
        print(f"[train] starting — max_steps={args.max_steps}  grad_accum={args.grad_accum}")

    while step < args.max_steps:
        # LR schedule
        cur_lr = lr_schedule(step, args.warmup_steps, args.max_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        for micro_step in range(args.grad_accum):
            # Fetch batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch.to(device)  # [B, seq_len]
            B, seq_len = input_ids.shape

            if seq_len <= args.block_size + 2:
                continue  # sequence too short — skip

            # Sample block position
            block_start, block_end = build_mask_schedule(args.block_size, seq_len)
            # Ground truth tokens for this block
            block_ids = input_ids[:, block_start:block_end].clone()  # [B, block_size]

            # Build masked input: replace block with MASK token
            masked_ids = input_ids.clone()
            masked_ids[:, block_start:block_end] = args.mask_token_id

            # Run frozen target on masked sequence
            with torch.no_grad():
                target_features, target_logits = run_target(
                    target, masked_ids, draft_inner.target_layer_ids
                )
                # Context features: everything BEFORE the block (positions 0..block_start-1).
                # This is what the draft conditions on — the unmasked prefix.
                ctx_target_feats    = target_features[:, :block_start, :]            # [B, block_start, n*H]
                block_target_logits = target_logits[:, block_start:block_end, :]     # [B, bs, V]

            # Embed masked block tokens (MASK embeddings)
            with torch.no_grad():
                noise_emb = target.model.embed_tokens(masked_ids[:, block_start:block_end])  # [B, bs, H]

            # Position ids: [0..block_end-1] covering ctx (0..block_start-1) then block
            # The model splits this into ctx_positions and block_positions internally.
            full_pos_ids = torch.arange(block_end, device=device).unsqueeze(0).expand(B, -1)  # [B, block_end]

            # DFlash draft forward
            draft_hidden = draft(
                noise_embedding=noise_emb,
                target_features=ctx_target_feats,
                position_ids=full_pos_ids,
            )  # [B, block_size, H]

            draft_logits = target.lm_head(draft_hidden)  # [B, block_size, V]

            loss = compute_loss(
                draft_logits=draft_logits.float(),
                target_ids=block_ids,
                target_logits=block_target_logits.float(),
                alpha_ce=args.alpha_ce,
                alpha_kl=args.alpha_kl,
                temperature=args.kl_temperature,
            )

            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            accum_loss += loss.item() / args.grad_accum
            # Track CE separately for logging
            with torch.no_grad():
                V = draft_logits.shape[-1]
                ce = F.cross_entropy(draft_logits.view(-1, V).float(), block_ids.view(-1)).item()
                accum_ce += ce / args.grad_accum

        # Gradient step
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(trainable_params, args.clip_grad)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        # Logging
        if is_main and step % args.log_every == 0:
            print(
                f"[step {step:5d}/{args.max_steps}]  "
                f"loss={accum_loss:.4f}  ce={accum_ce:.4f}  lr={cur_lr:.2e}"
            )
            if args.wandb:
                import wandb
                wandb.log({"loss": accum_loss, "ce": accum_ce, "lr": cur_lr}, step=step)
        accum_loss = 0.0
        accum_ce = 0.0
        accum_kl = 0.0

        # Checkpoint — write new, then delete previous to keep only one on disk
        if is_main and step % args.save_every == 0:
            save_draft(draft_inner, args.out_dir, step, target_config, args, prev_step=last_saved_step)
            last_saved_step = step

    # Final save
    if is_main:
        save_draft(draft_inner, args.out_dir, step, target_config, args, prev_step=last_saved_step)
        print("[train] done.")

    teardown_distributed()


if __name__ == "__main__":
    main()
