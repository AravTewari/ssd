# Diffusion-as-SSD Final Wrap-Up

This document is the repo entrypoint for the final evaluation pass on the diffusion-as-SSD-drafter project.

The project is in evaluation-only mode:

- no new diffusion drafter backends
- no more predictor retraining
- no more branch-policy tuning
- no more EAGLE future-feature work

The final deliverable is a matched-budget negative-result package showing:

- AR vs diffusion-family oracle ceilings
- AR vs diffusion-family realized systems
- one longer-output regime that should favor SSD more than the short-output runs
- uncertainty bars on throughput and accepted suffix

## Main Driver

Run [eval_final_wrapup.py](/Users/arav/Dev/cmu/18789/ssd/scripts/eval_final_wrapup.py).

To regenerate the publication figures from an existing artifact bundle without rerunning
the full benchmarks, run
[render_final_wrapup_figures.py](/Users/arav/Dev/cmu/18789/ssd/scripts/render_final_wrapup_figures.py).

Required inputs:

- `--target`: `Qwen3-8B` snapshot
- `--training-metadata`: held-out split metadata JSON
- `--dflash-draft`: `Qwen3-8B-DFlash-b16` snapshot
- `--dflash-predictor`: predictor checkpoint

Optional AR draft candidates:

- `--ar-draft-candidates qwen0.6b=/path/to/qwen0.6,llama1b=/path/to/llama1b`

If `--ar-draft-candidates` is omitted, the script looks for:

- `Qwen3-0.6B`
- `Llama-3.2-1B-Instruct`

under `SSD_HF_CACHE`, keeps the available ones, and records missing candidates in the final setup table.

Example command on `catalyst`:

```bash
cd /raid/user_data/aravt/ssd
source .venv/bin/activate

export SSD_HF_CACHE=/raid/catalyst/models/hub
export SSD_DATASET_DIR=/raid/user_data/aravt/datasets

python -O scripts/eval_final_wrapup.py \
  --target /raid/catalyst/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --training-metadata /raid/user_data/aravt/ssd/artifacts/dflash_predictor_main/training_metadata.json \
  --dflash-draft /raid/catalyst/models/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/5adfac36a234741b344cd906ca3fc6a94d7d5955 \
  --dflash-predictor /raid/catalyst/models/hub/dflash-predictors/qwen3-8b-dflash-b16 \
  --output-dir /raid/user_data/aravt/ssd/artifacts/final_wrapup
```

Use `--reuse-existing` to rebuild the final report and figures without rerunning the underlying matrices when the summary JSONs already exist.

## Generated Artifacts

The wrap-up script writes one combined artifact directory containing:

- `final_summary.json`
- `final_tables.md`
- `figure_oracle_ceiling.png`
- `figure_oracle_ceiling.pdf`
- `figure_normalized_speedup.png`
- `figure_normalized_speedup.pdf`
- `figure_budget_frontier.png`
- `figure_budget_frontier.pdf`
- `figure_error_bars.png`
- `figure_oracle_ceiling.svg`
- `figure_normalized_speedup.svg`
- `figure_budget_frontier.svg`
- `figure_dflash_branch_cache_failure.png`
- `figure_dflash_branch_cache_failure.pdf`
- `figure_dflash_branch_cache_failure.svg`
- `appendix_dflash_diagnostics.md`
- `appendix_extra_regime.md`

It also writes the raw sub-run summaries under:

- `runs/ar/...`
- `runs/dflash/...`
- `runs/ddtree/...`

## What The Final Bundle Should Establish

The final recommendation should only claim what the measured regimes support.

If the output matches the current expectation, the conclusion should be:

- AR remains the better SSD drafter family on the tested Qwen3-8B + 2x B200 setup
- DFlash and DDTree do not beat AR on oracle throughput
- diffusion-family oracle modes also fail to exceed their own `exact_off` baselines by a meaningful margin

At that point, the project is done. The result is a negative result, not an unfinished implementation.
