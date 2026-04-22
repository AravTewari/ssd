#!/usr/bin/env bash
set -euo pipefail

remote_root=/raid/user_data/aravt/ssd
target=/raid/catalyst/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
draft=/raid/catalyst/models/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/5adfac36a234741b344cd906ca3fc6a94d7d5955
predictor=/raid/user_data/aravt/ssd/artifacts/dummy_dflash_predictor
meta=/raid/user_data/aravt/ssd/artifacts/ddtree_prompt_split/training_metadata.json
root=/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/dflash/main/summary_artifacts
out=/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/dflash/main/summary.json

port=46000

for batch in 1 2 4; do
  for mode in exact_dflash dflash_ssd_exact_off_normal dflash_ssd_exact_on_oracle dflash_ssd_predicted_off_oracle dflash_ssd_predicted_on_oracle dflash_ssd_predicted_on_normal; do
    adir="$root/matrix_a/${mode}_b${batch}"
    if ssh catalyst "[ -f '$adir/result.json' ]"; then
      echo "SKIP matrix_a mode=$mode b=$batch"
    else
      echo "RUN matrix_a mode=$mode b=$batch port=$port"
      ssh catalyst "cd $remote_root && export PATH=$remote_root/.venv/bin:/usr/local/bin:/usr/bin:/bin SSD_HF_CACHE=/raid/catalyst/models/hub SSD_DATASET_DIR=/raid/user_data/aravt/datasets CUDA_VISIBLE_DEVICES=4,5 SSD_DIST_PORT=$port; ./.venv/bin/python -O scripts/eval_dflash_predictor.py --target '$target' --draft '$draft' --predictor '$predictor' --training-metadata '$meta' --output-len 32 --gpus 2 --max-prompts 78 --topk 10 --fanout-template-name baseline48 --mode '$mode' --batch-size '$batch' --artifact-dir '$adir'"
    fi
    port=$((port + 1))
  done
done

for template in baseline48 front1 front2 front4 front8 front12 front16 top4x2 top4x3 top4x4; do
  for batch in 1 2 4; do
    adir="$root/matrix_b/${template}_b${batch}"
    if ssh catalyst "[ -f '$adir/result.json' ]"; then
      echo "SKIP matrix_b template=$template b=$batch"
    else
      echo "RUN matrix_b template=$template b=$batch port=$port"
      ssh catalyst "cd $remote_root && export PATH=$remote_root/.venv/bin:/usr/local/bin:/usr/bin:/bin SSD_HF_CACHE=/raid/catalyst/models/hub SSD_DATASET_DIR=/raid/user_data/aravt/datasets CUDA_VISIBLE_DEVICES=4,5 SSD_DIST_PORT=$port; ./.venv/bin/python -O scripts/eval_dflash_predictor.py --target '$target' --draft '$draft' --predictor '$predictor' --training-metadata '$meta' --output-len 32 --gpus 2 --max-prompts 78 --topk 10 --fanout-template-name '$template' --mode dflash_ssd_predicted_on_normal --batch-size '$batch' --artifact-dir '$adir'"
    fi
    port=$((port + 1))
  done
done

quality_dir="$root/quality_metrics"
if ssh catalyst "[ -f '$quality_dir/result.json' ]"; then
  echo "SKIP quality_metrics"
else
  echo "RUN quality_metrics port=$port"
  ssh catalyst "cd $remote_root && export PATH=$remote_root/.venv/bin:/usr/local/bin:/usr/bin:/bin SSD_HF_CACHE=/raid/catalyst/models/hub SSD_DATASET_DIR=/raid/user_data/aravt/datasets CUDA_VISIBLE_DEVICES=4,5 SSD_DIST_PORT=$port; ./.venv/bin/python -O scripts/eval_dflash_predictor.py --target '$target' --draft '$draft' --predictor '$predictor' --training-metadata '$meta' --output-len 32 --gpus 2 --max-prompts 78 --topk 10 --fanout-template-name baseline48 --mode quality_metrics --artifact-dir '$quality_dir'"
fi

ssh catalyst "cd $remote_root && ./.venv/bin/python - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, '/raid/user_data/aravt/ssd')
from scripts import eval_dflash_predictor as mod

root = Path('/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/dflash/main/summary_artifacts')
out = Path('/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/dflash/main/summary.json')
meta = '/raid/user_data/aravt/ssd/artifacts/ddtree_prompt_split/training_metadata.json'
predictor = '/raid/user_data/aravt/ssd/artifacts/dummy_dflash_predictor'
batch_sizes = [1, 2, 4]
matrix_a_modes = [
    'exact_dflash',
    'dflash_ssd_exact_off_normal',
    'dflash_ssd_exact_on_oracle',
    'dflash_ssd_predicted_off_oracle',
    'dflash_ssd_predicted_on_oracle',
    'dflash_ssd_predicted_on_normal',
]
matrix_b_templates = [
    'baseline48',
    'front1',
    'front2',
    'front4',
    'front8',
    'front12',
    'front16',
    'top4x2',
    'top4x3',
    'top4x4',
]

matrix_a_results = []
for batch in batch_sizes:
    for mode in matrix_a_modes:
        path = root / 'matrix_a' / f'{mode}_b{batch}' / 'result.json'
        if not path.exists():
            raise SystemExit(f'missing {path}')
        matrix_a_results.append(json.loads(path.read_text()))

matrix_b_results = []
for template in matrix_b_templates:
    for batch in batch_sizes:
        path = root / 'matrix_b' / f'{template}_b{batch}' / 'result.json'
        if not path.exists():
            raise SystemExit(f'missing {path}')
        matrix_b_results.append(json.loads(path.read_text()))

quality_path = root / 'quality_metrics' / 'result.json'
quality_metrics = json.loads(quality_path.read_text()) if quality_path.exists() else None
if quality_metrics is not None:
    exact_b1 = next(
        row for row in matrix_a_results
        if row['mode'] == 'dflash_ssd_exact_off_normal' and row['batch_size'] == 1
    )
    predicted_b1 = next(
        row for row in matrix_a_results
        if row['mode'] == 'dflash_ssd_predicted_off_oracle' and row['batch_size'] == 1
    )
    exact_accept = exact_b1['accepted_suffix_mean']
    predicted_accept = predicted_b1['accepted_suffix_mean']
    quality_metrics['accepted_suffix_exact_context'] = exact_accept
    quality_metrics['accepted_suffix_predicted_context'] = predicted_accept
    quality_metrics['accepted_suffix_delta'] = (
        None if exact_accept is None or predicted_accept is None else exact_accept - predicted_accept
    )

recommendation = mod._build_recommendation(matrix_a_results, matrix_b_results)
combined_table = mod._build_combined_markdown_table(matrix_a_results, matrix_b_results)
matrix_a_summary = {
    'results': matrix_a_results,
    'quality_metrics': quality_metrics,
    'recommendation': recommendation,
}
matrix_b_summary = {
    'results': matrix_b_results,
    'best_templates_by_batch': recommendation['decision_rule_c']['best_templates_by_batch'],
}
summary = {
    'training_metadata': meta,
    'predictor': predictor,
    'matrix_a': matrix_a_summary,
    'matrix_b': matrix_b_summary,
    'quality_metrics': quality_metrics,
    'recommendation': recommendation,
    'combined_markdown_table': combined_table,
}

out.parent.mkdir(parents=True, exist_ok=True)
(root / 'matrix_a_summary.json').write_text(json.dumps(matrix_a_summary, indent=2, sort_keys=True))
(root / 'matrix_b_summary.json').write_text(json.dumps(matrix_b_summary, indent=2, sort_keys=True))
out.write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f'WROTE {out}')
PY"
