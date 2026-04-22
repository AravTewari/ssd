# modal_bench

Run SSD benchmarks on Modal cloud GPUs.

## Setup

Modal token already wired into `~/.modal.toml` (profile `xinyuj2`).

## Quick run — AR baseline (TP=2, Qwen3-8B, GSM8K)

```bash
modal run modal_bench/ar_bench.py
```

First call builds the image (~5 min) and downloads Qwen3-8B into a Modal
Volume called `ssd-hf-cache` (~16 GB, one-time). Subsequent runs reuse both.

The benchmark prints full stdout plus a `KEY METRICS` block with
prefill / decode / end-to-end throughput.

## Notes

- GPU: `H200:2` — matches the 2 target TP GPU topology of the reference table.
- Dataset: `gsm8k_data_10000.jsonl` is copied into the image at build time
  from `/sgl-workspace/dgm/processed_datasets/gsm8k/`.
- `SSD_HF_CACHE=/cache/hf` lives on the shared volume so the model is
  downloaded once and reused.
