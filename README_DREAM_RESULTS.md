# Dream Diffusion Draft Results

This document summarizes the first end-to-end benchmark results for using a synchronous Dream diffusion model as the draft model in this speculative decoding engine.

The intended audience is someone already familiar with speculative decoding systems and interested in advising on the next research step.

## Executive Summary

We implemented a new `dream_diffusion` backend that uses Dream's native `diffusion_generate()` path and captures logits through `generation_logits_hook_func`, then benchmarks it against the existing synchronous autoregressive draft path.

On the tested setup, the diffusion draft is substantially slower than the autoregressive draft baseline.

Main outcome:

| Draft backend | Best tested config | Best throughput |
| --- | --- | ---: |
| Dream diffusion | `b=8`, `dsteps=8` | `57.72 tok/s` |
| Dream diffusion | `b=8`, `dsteps=16` | `37.30 tok/s` |
| AR baseline | `b=8`, `Qwen3-8B` draft | `342.48 tok/s` |

Key conclusion:

- Against the best Dream point we tested, the autoregressive baseline is about `5.9x` faster.
- Against Dream at `dsteps=16`, the autoregressive baseline is about `9.2x` faster.
- The Dream draft also has much lower acceptance: about `1.63` accepted tokens per step including recovery, versus `3.95` for the autoregressive baseline.

In this codebase and on this workload, the diffusion draft currently loses on both:

- draft-side latency
- verifier acceptance quality

## What Was Implemented

The following pieces were added or changed to support the Dream experiment:

- New `dream_diffusion` backend in [bench.py](/Users/arav/Dev/cmu/18789/ssd/bench/bench.py), [config.py](/Users/arav/Dev/cmu/18789/ssd/ssd/config.py), and [llm_engine.py](/Users/arav/Dev/cmu/18789/ssd/ssd/engine/llm_engine.py).
- New Dream adapter in [dream_diffusion_adapter.py](/Users/arav/Dev/cmu/18789/ssd/ssd/engine/dream_diffusion_adapter.py).
- Shared sync diffusion speculator path in [speculator_sync_diffusion.py](/Users/arav/Dev/cmu/18789/ssd/ssd/engine/speculator_sync_diffusion.py).
- GPU smoke test in [smoke_dream_diffusion.py](/Users/arav/Dev/cmu/18789/ssd/scripts/smoke_dream_diffusion.py).
- Benchmark sweep fix in [bench.py](/Users/arav/Dev/cmu/18789/ssd/bench/bench.py): initialize at the maximum requested sweep batch size so verify CUDA graphs exist for all swept `b`.
- B200 attention fix in [attention.py](/Users/arav/Dev/cmu/18789/ssd/ssd/layers/attention.py): choose FA4 on SM100+ instead of forcing FA3.
- Distributed bootstrap fix in [model_runner.py](/Users/arav/Dev/cmu/18789/ssd/ssd/engine/model_runner.py): read `SSD_DIST_PORT` from the environment to avoid repeated port collisions on `catalyst`.

## Environment

All reported results below were run on `catalyst`.

Hardware and runtime:

- GPUs: NVIDIA B200
- Target parallelism: 2-GPU tensor parallel
- Visible devices used for successful runs: `CUDA_VISIBLE_DEVICES=6,7`
- Python env: `/raid/user_data/aravt/ssd/.venv`
- Repo path on server: `/raid/user_data/aravt/ssd`

Model paths:

- Target: `Qwen3-32B`
- Dream draft: `Dream-v0-Instruct-7B`
- AR draft baseline: `Qwen3-8B`

Important caveat:

- The original desired AR baseline was `Qwen3-0.6B`, but that checkpoint was not present on `catalyst`.
- We therefore used `Qwen3-8B` as the autoregressive draft baseline.
- This is not the same baseline as the original plan, but it is a more size-matched comparison to `Dream-7B` than `0.6B` would have been.

## Compatibility Notes

The Dream path was chosen because it is token-compatible with the Qwen target in practice.

What was validated:

- normal text tokenization matches the target tokenizer
- important special-token ids align by token string
- Dream logits can be captured and returned in the `[B, K, V]` format the verifier expects

One subtlety came up during validation:

- Dream does not expose the same default `eos_token_id` choice as the target config
- however, the special-token mapping is aligned by token string and token id
- the compatibility check was updated to validate actual token alignment, not identical default EOS selection

## Benchmark Methodology

Unless otherwise noted, all runs used:

| Setting | Value |
| --- | --- |
| Target model | `Qwen3-32B` |
| Speculation | enabled |
| `k` | `6` |
| Decoding mode | greedy |
| Prompts | random token prompts |
| `numseqs` | `16` |
| `input_len` | `32` |
| `output_len` | `32` |
| Target GPUs | `2` |
| CUDA graphs | enabled |

For Dream:

- sync-only diffusion draft
- no async path
- no stochastic sampling
- measured `dsteps` sweep at fixed `b=8`

For AR:

- standard synchronous speculative decoding path already present in the engine
- draft model set to `Qwen3-8B`

## Dream Diffusion Results

### Batch sweep at fixed `dsteps=16`

Command shape:

```bash
python -O bench.py \
  --qwen --size 32 --gpus 2 \
  --spec --draft-backend dream_diffusion \
  --draft /raid/user_data/aravt/huggingface/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae \
  --numseqs 16 --input_len 32 --output_len 32 \
  --random --temp 0 --dsteps 16 \
  --sweep '[{"b":1},{"b":2},{"b":4},{"b":8}]'
```

Results:

| `b` | Throughput | Mean accepted suffix incl. recovery | Mean target step | Mean target verify | Mean diffusion draft step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1` | `6.20 tok/s` | `1.61` | `254.02 ms` | `12.04 ms` | `239.86 ms` |
| `2` | `11.36 tok/s` | `1.60` | `265.17 ms` | `11.58 ms` | `254.22 ms` |
| `4` | `21.45 tok/s` | `1.62` | `268.18 ms` | `11.66 ms` | `258.28 ms` |
| `8` | `37.30 tok/s` | `1.63` | `274.49 ms` | `12.09 ms` | `266.03 ms` |

Observed `b*` on the tested grid:

- `b=8`

Interpretation:

- Throughput increased monotonically over the tested batch range.
- Acceptance did not meaningfully change with `b`.
- Diffusion draft latency increased slightly with larger batch size, but not enough to offset batching gains.

### Denoising-step sweep at fixed `b=8`

Command shape:

```bash
python -O bench.py \
  --qwen --size 32 --gpus 2 \
  --spec --draft-backend dream_diffusion \
  --draft /raid/user_data/aravt/huggingface/models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae \
  --b 8 --numseqs 16 --input_len 32 --output_len 32 \
  --random --temp 0 --dsteps 16 \
  --sweep '[{"dsteps":8},{"dsteps":16},{"dsteps":32},{"dsteps":64}]'
```

Results:

| `dsteps` | Throughput | Mean accepted suffix incl. recovery | Mean target step | Mean target verify | Mean diffusion draft step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `8` | `57.72 tok/s` | `1.63` | `177.36 ms` | `17.31 ms` | `150.20 ms` |
| `16` | `38.50 tok/s` | `1.63` | `265.94 ms` | `14.09 ms` | `255.23 ms` |
| `32` | `19.91 tok/s` | `1.63` | `514.20 ms` | `16.60 ms` | `506.01 ms` |
| `64` | `9.98 tok/s` | `1.63` | `1026.31 ms` | `24.95 ms` | `1020.14 ms` |

Observed best Dream point:

- `b=8`, `dsteps=8`
- throughput `57.72 tok/s`

Interpretation:

- On this workload, increasing denoising depth did not improve acceptance.
- Throughput degraded almost exactly as draft latency increased.
- This suggests Dream is not currently trading extra denoising compute for meaningfully better speculative proposals in this setup.

## Autoregressive Baseline Results

### Batch sweep using `Qwen3-8B` draft

Command shape:

```bash
python -O bench.py \
  --qwen --size 32 --gpus 2 \
  --spec --draft-backend ar \
  --draft /raid/catalyst/models/hub/models--Qwen--Qwen3-8B \
  --numseqs 16 --input_len 32 --output_len 32 \
  --random --temp 0 \
  --sweep '[{"b":1},{"b":2},{"b":4},{"b":8}]'
```

Results:

| `b` | Throughput | Mean accepted suffix incl. recovery | Mean target step | Mean target verify |
| --- | ---: | ---: | ---: | ---: |
| `1` | `55.98 tok/s` | `3.95` | `65.30 ms` | `22.71 ms` |
| `2` | `135.02 tok/s` | `3.95` | `52.63 ms` | `17.38 ms` |
| `4` | `236.08 tok/s` | `3.95` | `57.02 ms` | `20.70 ms` |
| `8` | `342.48 tok/s` | `3.95` | `59.74 ms` | `21.95 ms` |

Observed `b*` on the tested grid:

- `b=8`

Interpretation:

- The AR draft scales very well with batch size on this setup.
- Acceptance is dramatically better than Dream at the same `k`.
- The target verify cost is not the bottleneck here; the quality and speed of the draft model dominate the result.

## Direct Comparison

### Best tested Dream vs AR baseline

| Metric | Dream best | AR baseline best | Ratio |
| --- | ---: | ---: | ---: |
| Throughput | `57.72 tok/s` | `342.48 tok/s` | AR is `5.93x` faster |
| Accepted suffix incl. recovery | `1.63` | `3.95` | AR is `2.42x` higher |

### Dream at `dsteps=16` vs AR baseline

| Metric | Dream `dsteps=16`, `b=8` | AR `b=8` | Ratio |
| --- | ---: | ---: | ---: |
| Throughput | `37.30 tok/s` | `342.48 tok/s` | AR is `9.18x` faster |
| Accepted suffix incl. recovery | `1.63` | `3.95` | AR is `2.42x` higher |

### Shape of the result

The most important structural observation is not just that Dream is slower. It is slower for two independent reasons:

1. Draft generation is slower.
2. The verifier accepts far fewer speculative tokens.

That combination is what makes the current Dream path uncompetitive.

## Practical Interpretation

The current Dream backend is a useful systems experiment, but it is not yet a good speculative draft model for this engine.

The data suggests:

- The current diffusion draft is not generating high-quality enough speculative continuations per unit time.
- Increasing denoising depth does not rescue acceptance.
- The system already batches Dream reasonably well, so the remaining problem appears to be model-side draft efficiency and proposal quality, not merely scheduler underutilization.

Put differently:

- if the goal is immediate throughput gains, the AR draft is clearly better
- if the goal is research on diffusion drafting, the problem now looks like a model-quality or interface problem, not a benchmark artifact

## Limitations of This Comparison

These caveats matter:

- The AR baseline is `Qwen3-8B`, not `Qwen3-0.6B`.
- Dream uses native diffusion logits in a greedy-only sync setup.
- We did not attempt any Dream-specific acceptance-oriented tuning beyond the denoising-step sweep.
- We did not tune `k` separately for Dream and AR.
- We did not test async speculation for Dream.
- We did not test larger batch sizes than `8`.

So these numbers should be read as:

- a valid first systems comparison
- not a final statement that diffusion drafting cannot work
- but a strong indication that this specific Dream integration is not currently competitive

## Questions for an Expert Reviewer

The following are the most useful questions to answer next:

1. Is there a better way to derive verifier-compatible logits from Dream than using the final denoising-step logits directly?
2. Is `k=6` a poor operating point for Dream, even though it works well for AR?
3. Should Dream be evaluated with a different notion of draft confidence than the current greedy verifier interface exposes?
4. Is there a better denoising schedule or truncation strategy that would improve acceptance materially without collapsing throughput?
5. Is sync speculative decoding fundamentally the wrong integration point for a diffusion drafter, meaning the next experiment should be async or tree-based rather than more tuning of the sync path?
6. Is `Qwen3-8B` the right AR baseline for this comparison, or should we spend the time to fetch `Qwen3-0.6B` and compare against both a small and size-matched AR draft?

## Recommended Next Steps

Based on the data so far, the most reasonable next steps are:

1. Fetch `Qwen3-0.6B` and run the same AR sweep so we have both a small-draft and size-matched-draft AR baseline.
2. Sweep `k` for Dream and AR separately, because Dream may want a different speculative horizon.
3. Investigate whether Dream's final logits are the wrong object for verifier acceptance, even if they are easy to extract.
4. If the intent is to make diffusion competitive as a system, consider an async design rather than pushing further on the current sync path.

## Bottom Line

Current status:

- The Dream integration works.
- The smoke test passes.
- The B200 benchmark path works.
- The sweep path works.
- The performance result is decisively negative relative to the autoregressive baseline.

That is still a useful outcome. The implementation and measurements are good enough now to ask a domain expert a sharper question:

- should we keep tuning a synchronous diffusion draft path, or is the next serious idea an interface change rather than a parameter change?
