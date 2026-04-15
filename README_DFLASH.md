# DFlash: Block Diffusion for Speculative Decoding

DFlash trains a lightweight block-diffusion draft model conditioned on the target model's intermediate hidden states. At inference the draft predicts a block of tokens in parallel; the target verifies them with cascaded acceptance (accept the longest correct prefix).

Based on [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) and [BlockDiffusion](https://arxiv.org/abs/2503.09573).

---

## Files

```
DFlash/
  __init__.py
  dflash_model.py     # DFlashDraftModel architecture
  train.py            # Training script for Qwen3-32B
  dflash_adapter.py   # Inference adapter for the SSD engine
```

---

## Architecture

DFlash makes the following key modifications to BlockDiffusion:

- **Cross-attention to target hidden states.** Each draft attention layer attends over two KV sets: projected intermediate hidden states from the target (tapped at evenly-spaced layers), and the noised block embedding itself. Queries come only from the noised block.
- **Shared embed/lm_head.** The draft reuses the target's `embed_tokens` and `lm_head` — no extra vocab projection.
- **Non-causal block attention.** All block positions attend to each other and to the full context in parallel (`is_causal=False`).
- **KV-cache crop.** After each speculative step the draft KV-cache is cropped to the accepted prefix length, keeping memory O(1) per step.

The draft has `num_draft_layers` transformer layers (1–3 in the paper) and the same hidden size as the target, so no dimension mismatch.

---

## Training

### Prerequisites

- 8× B200 (or H100/H200) GPUs — the 32B target runs frozen in bf16 (~64 GB)
- A processed dataset directory (`$SSD_DATASET_DIR`) with `.jsonl` files
- Qwen3-32B model weights

### Launch

```bash
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
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--target` | required | Path to Qwen3-32B model directory |
| `--out_dir` | required | Where to save draft checkpoints |
| `--dataset` | `alpaca` | Dataset name (`alpaca`, `gsm`, `humaneval`, `ultrafeedback`, `c4`) or path to `.jsonl` dir |
| `--dataset_dir` | `$SSD_DATASET_DIR` | Root of processed datasets |
| `--num_draft_layers` | `1` | Number of draft transformer layers |
| `--block_size` | `4` | Tokens predicted in parallel per step |
| `--mask_token_id` | `151666` | Qwen3 mask token id |
| `--max_seq_len` | `4096` | Training context length |
| `--batch_size` | `1` | Per-GPU batch size |
| `--grad_accum` | `8` | Gradient accumulation steps |
| `--max_steps` | `5000` | Total optimizer steps |
| `--lr` | `1e-4` | Peak learning rate (cosine decay with warmup) |
| `--alpha_ce` | `1.0` | Cross-entropy loss weight |
| `--alpha_kl` | `1.0` | KL distillation loss weight |
| `--wandb` | off | Enable Weights & Biases logging |

### Training objective

For each training sequence, a contiguous block of `block_size` tokens is randomly sampled and replaced with the MASK token. The draft model receives:

1. The target model's intermediate hidden states (tapped at `num_draft_layers` evenly-spaced layers) for the masked sequence.
2. The token embeddings of the masked block (all MASK tokens).

Loss = CE (hard labels on the original tokens) + KL divergence (soft distillation from the target's output distribution, temperature-scaled).

Only the draft's new parameters are trained — the target stays frozen throughout.

### Checkpoints

Each checkpoint is saved under `out_dir/step_N/` and contains:

```
step_N/
  pytorch_model.bin    # draft weights (excludes shared embed/lm_head)
  dflash_config.json   # hyperparameters needed to reconstruct the model
```

---

## Inference

Point `--draft` at a DFlash checkpoint directory. The engine automatically detects the `dflash_config.json` and loads the draft.

### Python API

```python
from ssd.llm import LLM
from ssd.sampling_params import SamplingParams

llm = LLM(
    model="/path/to/Qwen3-32B",
    draft="/path/to/dflash_qwen3_32b/step_5000",
    speculate=True,
    draft_backend="dflash",
    speculate_k=4,
)

outputs = llm.generate(
    ["Explain speculative decoding in one sentence."],
    SamplingParams(temperature=0.0, max_new_tokens=256),
)
print(outputs[0])
```

### Environment variables

| Variable | Description |
|---|---|
| `SSD_DFLASH_DRAFT_MODEL` | Default path to a trained DFlash checkpoint |
| `SSD_TARGET_MODEL` | Default path to the Qwen3-32B target |
| `SSD_HF_CACHE` | HuggingFace cache hub directory |
| `SSD_DATASET_DIR` | Root directory for processed training datasets |

---

## Notes

- `draft_backend="dflash"` only supports **greedy decoding** (`temperature=0`) for now, matching the DFlash paper's evaluation setting.
- The adapter loads a separate frozen HuggingFace copy of the target for hidden-state extraction. On a single B200 (192 GB HBM3e) a 32B bf16 model (~64 GB) fits comfortably alongside the draft.
- FlashAttention-3 is used automatically on B200 (SM100) via `F.scaled_dot_product_attention` with PyTorch ≥ 2.4.
