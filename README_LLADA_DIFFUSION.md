# LLaDA Diffusion Draft Backend

This document explains the diffusion-draft changes added to this repository, why they were made, and how to use them for the first round of experiments.

## What Changed

We added a new speculative draft backend:

- `draft_backend="ar"`: the original autoregressive draft path
- `draft_backend="llada_diffusion"`: a new synchronous diffusion draft path based on LLaDA

The new backend is wired into:

- engine config in [`ssd/config.py`](/Users/arav/Dev/cmu/18789/ssd/ssd/config.py)
- engine construction in [`ssd/engine/llm_engine.py`](/Users/arav/Dev/cmu/18789/ssd/ssd/engine/llm_engine.py)
- a standalone diffusion adapter in [`ssd/engine/diffusion_draft_adapter.py`](/Users/arav/Dev/cmu/18789/ssd/ssd/engine/diffusion_draft_adapter.py)
- a dedicated sync diffusion speculator in [`ssd/engine/speculator_sync_diffusion.py`](/Users/arav/Dev/cmu/18789/ssd/ssd/engine/speculator_sync_diffusion.py)
- benchmark CLI and sweep reporting in [`bench/bench.py`](/Users/arav/Dev/cmu/18789/ssd/bench/bench.py)
- draft path resolution in [`bench/bench_helpers.py`](/Users/arav/Dev/cmu/18789/ssd/bench/bench_helpers.py)
- a smoke test script in [`scripts/smoke_llada_diffusion.py`](/Users/arav/Dev/cmu/18789/ssd/scripts/smoke_llada_diffusion.py)

## Why We Made These Changes

The goal is to run an experiment where the speculative draft model is no longer autoregressive. Instead, we use a diffusion model to propose the next `K` tokens and then compare:

- throughput against the autoregressive draft baseline
- acceptance behavior under target verification
- how the optimal batch size `b*` shifts when the draft model changes

The key requirement for this experiment is that the draft path must still provide `logits_q`, because the existing verifier consumes:

- speculative draft tokens
- draft logits with shape `[B, K, V]`

That requirement made LLaDA the best v1 candidate because its generation loop already exposes token logits directly, which fits the current verifier contract with much less surgery than adding a separate trainable output head.

## Main Design Decisions

### 1. Use LLaDA native logits

We chose to use native logits from the diffusion model instead of attaching a trainable head.

Reason:

- it keeps the experiment closer to the actual diffusion model behavior
- it avoids introducing a second training problem before measuring inference behavior
- it keeps the first prototype small enough to benchmark quickly

### 2. Keep the target path unchanged

The target model, scheduler, and verifier were already stable. We did not route the diffusion draft through `ModelRunner`.

Instead, we added a separate adapter that:

- loads LLaDA independently with `trust_remote_code=True`
- runs a diffusion fill for the next `K` positions
- returns:
  - drafted tokens of shape `[B, K]`
  - draft logits of shape `[B, K, V]`

This preserves the existing target verification path and reduces integration risk.

### 3. Add a separate sync diffusion speculator

We did not overload the existing autoregressive sync speculator.

Reason:

- the autoregressive draft path advances token-by-token through `ModelRunner`
- the diffusion draft path fills an entire speculative block in one call

Those are different execution models, so a dedicated `SpeculatorSyncDiffusion` keeps the code easier to reason about.

### 4. Constrain v1 heavily

The first implementation is intentionally narrow:

- synchronous speculative decoding only
- greedy decoding only
- Qwen target only
- same tokenizer and vocab required between target and diffusion draft

These constraints were chosen because the first objective is to measure speed and `b*`, not to generalize the engine immediately.

## Why The Constraints Exist

### Sync only

The async SSD path is built around a draft-side tree cache and future verification outcomes. That design assumes an autoregressive draft model with cheap branch extension and cached logits.

A diffusion draft changes that cost model substantially. Rather than mix both problems together, we implemented only the synchronous version first.

### Greedy only

The existing verifier supports exact ratio-style acceptance when the draft distribution `q` is well-defined and sampled in the expected way.

For the diffusion experiment, we are currently using final denoising-step logits as a practical proxy for `q`. That is fine for greedy acceptance experiments, but we are not claiming an exact stochastic speculative-decoding interpretation in v1.

### Same tokenizer and vocab

The verifier expects draft logits in the same token space as the target.

To avoid adding a projection layer or token remapping system before the first benchmark pass, the adapter performs strict compatibility checks:

- same vocab size
- matching EOS/PAD semantics where relevant
- identical tokenization on a probe string
- diffusion mask id must not collide with target PAD

If those checks fail, the backend aborts early.

## How The Diffusion Draft Path Works

At each speculative step:

1. The target produces the recovery token as usual.
2. The sync diffusion speculator appends that recovery token to the sequence.
3. The LLaDA adapter fills the next `K` positions with diffusion generation.
4. The adapter returns:
   - the `K` drafted tokens
   - the final-step logits for those `K` positions
5. The existing verifier compares those proposals against the target model.

The returned `SpeculateResult` still has the same shape contract as the autoregressive path:

- `speculations`: `[B, K+1]`, where column 0 is the recovery token
- `logits_q`: `[B, K, V]`

That is why the rest of the speculative pipeline can stay mostly unchanged.

## Benchmarking Changes

We extended the benchmark entrypoint so diffusion runs can be measured the same way as the autoregressive baseline.

New benchmark options:

- `--draft-backend {ar,llada_diffusion}`
- `--dsteps`

The sweep flow now also records and prints:

- best batch size `b`
- best end-to-end throughput
- mean accepted suffix length
- mean diffusion draft step time when using the diffusion backend

This is aimed directly at the `b*` experiment.

## Smoke Test

We added a smoke test script:

- [`scripts/smoke_llada_diffusion.py`](/Users/arav/Dev/cmu/18789/ssd/scripts/smoke_llada_diffusion.py)

It validates:

- model loading
- tokenizer/vocab compatibility
- one tiny diffusion speculative fill
- output shapes for drafted tokens and logits

Example:

```bash
python3 scripts/smoke_llada_diffusion.py \
  --target /path/to/qwen/target/snapshot \
  --draft /path/to/llada/model \
  --b 1 \
  --k 4 \
  --dsteps 32
```

## Benchmark Example

Example diffusion-draft benchmark:

```bash
cd bench
python -O bench.py \
  --qwen \
  --size 32 \
  --spec \
  --draft-backend llada_diffusion \
  --draft /path/to/llada/model \
  --k 6 \
  --dsteps 128 \
  --b 1 \
  --temp 0 \
  --numseqs 128 \
  --output_len 512
```

Example batch-size sweep:

```bash
cd bench
python -O bench.py \
  --qwen \
  --size 32 \
  --spec \
  --draft-backend llada_diffusion \
  --draft /path/to/llada/model \
  --k 6 \
  --dsteps 128 \
  --temp 0 \
  --numseqs 128 \
  --output_len 512 \
  --sweep '[{"b":1},{"b":2},{"b":4},{"b":8}]'
```

Example diffusion-step sweep after choosing a provisional `b*`:

```bash
cd bench
python -O bench.py \
  --qwen \
  --size 32 \
  --spec \
  --draft-backend llada_diffusion \
  --draft /path/to/llada/model \
  --k 6 \
  --temp 0 \
  --numseqs 128 \
  --output_len 512 \
  --sweep '[{"b":4,"dsteps":32},{"b":4,"dsteps":64},{"b":4,"dsteps":128}]'
```

## Current Limitations

- no async diffusion draft path
- no support for stochastic diffusion speculative verification
- no cross-tokenizer or cross-vocab projection layer
- no Dream-7B backend yet
- no claim that this v1 diffusion mode is exact speculative decoding in the stochastic sense

## What To Look At Next

If the diffusion backend shows promising throughput or a better `b*`, the next follow-up questions are:

- how sensitive is throughput to `diffusion_steps`
- how much acceptance changes as denoising depth changes
- whether a projection layer or shared-vocab alternative would unlock more targets
- whether an async diffusion design is worth building at all
