# DFlash in SSD

This document summarizes the repo-native `dflash` backend added for exact regular speculative decoding with:

- target: `Qwen3-8B`
- draft: `z-lab/Qwen3-8B-DFlash-b16`

It also records the exact-feasibility result for DFlash in this repo's spec-spec architecture.

## What Was Added

The new backend name is `dflash`.

Key pieces:

- config and validation in `ssd/config.py`
- remote DFlash worker on GPU 1 in `ssd/engine/dflash_worker.py`
- sync DFlash speculator in `ssd/engine/speculator_sync_dflash.py`
- exact target-feature extraction for Qwen3 in `ssd/models/qwen3.py`
- verifier / scheduler plumbing for accepted DFlash target features in:
  - `ssd/engine/verifier.py`
  - `ssd/engine/scheduler.py`
  - `ssd/engine/sequence.py`
  - `ssd/engine/helpers/speculate_types.py`
- engine wiring and metrics in `ssd/engine/llm_engine.py`
- benchmark and path support in:
  - `bench/bench.py`
  - `bench/bench_helpers.py`
  - `bench/bench_paths.py`
  - `ssd/paths.py`
- smoke test in `scripts/smoke_dflash.py`

## Runtime Contract

`draft_backend="dflash"` currently enforces:

- speculative decoding enabled
- Qwen target family only
- exact sync-only mode
- greedy decoding only
- `num_gpus=2`
- dedicated target on GPU 0 and DFlash worker on GPU 1
- `speculate_k = block_size - 1`, derived from the DFlash draft config

The backend also forces:

- `enforce_eager=True`
- `kvcache_block_size=128`

The `kvcache_block_size` override is required on B200 / SM100 because `sgl-kernel` paged-KV verify currently only supports `page_size=128` there.

## Exact DFlash Data Flow

This implementation stays exact for regular speculative decoding.

Target side:

- the target prefill / verify runs on GPU 0 through the repo's Qwen3 path
- Qwen3 now optionally returns concatenated hidden features from the DFlash-selected target layers
- the verifier slices those exact features for:
  - prompt prefill
  - accepted verified suffixes

Draft side:

- the DFlash worker runs on GPU 1
- it loads:
  - the DFlash draft model
  - a HF `Qwen3-8B` target model for `embed_tokens` and `lm_head`
- it keeps per-sequence DFlash `DynamicCache` state
- each draft step uses:
  - the current recovery token
  - the exact accepted target features from the previous verify result

This matches the upstream DFlash conditioning pattern:

- cache exact prefix state
- condition the next block draft on freshly produced target hidden features

## Spec-Spec Feasibility Result

Exact DFlash spec-spec is explicitly blocked in this repo.

Reason:

- the next DFlash draft block depends on exact target hidden features that are only produced after the current target verify step completes
- those features are part of the true conditioning state for the next draft step
- therefore the SSD async cache key would have to include information that is not available pre-verify

Current behavior:

- `draft_backend=dflash` with `draft_async=True` fails at startup with:

`dflash exact async/spec-spec is unsupported because next-step drafting requires fresh target hidden features produced by the current verify step`

This is intentional. No hidden-state predictor or approximate substitute was added.

## Remote Validation

Validated on `catalyst` with B200 GPUs.

Environment:

- `SSD_HF_CACHE=/raid/catalyst/models/hub`
- `SSD_DATASET_DIR=/raid/user_data/aravt/datasets`
- repo: `/raid/user_data/aravt/ssd`
- venv: `/raid/user_data/aravt/ssd/.venv`

Models used:

- target: `/raid/catalyst/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`
- draft: `/raid/catalyst/models/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/5adfac36a234741b344cd906ca3fc6a94d7d5955`

### Smoke Test

Command:

```bash
export SSD_HF_CACHE=/raid/catalyst/models/hub
export SSD_DATASET_DIR=/raid/user_data/aravt/datasets
export CUDA_VISIBLE_DEVICES=1,2
export SSD_DIST_PORT=12363

cd /raid/user_data/aravt/ssd
source .venv/bin/activate

python scripts/smoke_dflash.py \
  --target /raid/catalyst/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --draft /raid/catalyst/models/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/5adfac36a234741b344cd906ca3fc6a94d7d5955 \
  --gpus 2 \
  --max-steps 2
```

Observed result:

| Check | Result |
| --- | --- |
| Smoke status | Pass |
| `k` | `15` |
| DFlash block size | `16` |
| Prompt feature shape | `(8, 20480)` |
| Speculations shape | `(1, 16)` |
| `logits_q` shape | `(1, 15, 151936)` |
| Mean DFlash draft step time | `647.76 ms` |

The smoke path also completed a real verify pass and returned:

- accepted suffix: `[10956, 22160, 47116]`
- next recovery token: `374`

### Tiny Regular-DFlash Benchmark

Command:

```bash
export SSD_HF_CACHE=/raid/catalyst/models/hub
export SSD_DATASET_DIR=/raid/user_data/aravt/datasets
export CUDA_VISIBLE_DEVICES=1,2
export SSD_DIST_PORT=12364

cd /raid/user_data/aravt/ssd/bench
source ../.venv/bin/activate

python -O bench.py \
  --qwen --size 8 \
  --spec \
  --draft-backend dflash \
  --draft /raid/catalyst/models/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/5adfac36a234741b344cd906ca3fc6a94d7d5955 \
  --gpus 2 \
  --b 1 \
  --numseqs 1 \
  --input_len 8 \
  --output_len 4 \
  --random \
  --temp 0 \
  --max-steps 2
```

Observed result:

| Metric | Value |
| --- | --- |
| Total throughput | `0.67 tok/s` |
| Avg tokens per step incl. recovery | `1.00` |
| Avg fraction of drafted tokens accepted | `0.00` |
| Avg target step time | `2845.21 ms` |
| Avg target verify time | `4328.49 ms` |
| Avg DFlash draft step time | `750.17 ms` |

This was only a tiny sanity benchmark, not a tuned performance sweep.

### Exact Async Rejection

Command:

```bash
export SSD_HF_CACHE=/raid/catalyst/models/hub
export SSD_DATASET_DIR=/raid/user_data/aravt/datasets
export CUDA_VISIBLE_DEVICES=1,2
export SSD_DIST_PORT=12365

cd /raid/user_data/aravt/ssd/bench
source ../.venv/bin/activate

python -O bench.py \
  --qwen --size 8 \
  --spec --async \
  --draft-backend dflash \
  --draft /raid/catalyst/models/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/5adfac36a234741b344cd906ca3fc6a94d7d5955 \
  --gpus 2 \
  --b 1 \
  --numseqs 1 \
  --input_len 8 \
  --output_len 4 \
  --random \
  --temp 0
```

Observed result:

| Mode | Result |
| --- | --- |
| Exact regular DFlash SD | Runs |
| Exact DFlash spec-spec | Blocked at startup |

Startup error:

`ValueError: dflash exact async/spec-spec is unsupported because next-step drafting requires fresh target hidden features produced by the current verify step`

## Current Bottom Line

What works:

- exact regular DFlash speculative decoding
- repo-native target-feature extraction
- remote DFlash worker on a second GPU
- smoke-tested and tiny-benchmark-tested on `catalyst`

What does not exist:

- exact DFlash spec-spec in this SSD architecture

Why not:

- exact next-step DFlash drafting depends on target hidden features that do not exist until after verify finishes

## Next Useful Work

If the goal is to compare DFlash to ordinary AR draft baselines:

- run proper `b` sweeps for regular DFlash
- compare to an AR draft under the same 2-GPU budget

If the goal is to revive spec-spec with DFlash:

- that requires changing the exactness requirement
- specifically, adding a predictor or approximate surrogate for future target hidden features
- that is a separate research direction, not a small extension of this implementation
