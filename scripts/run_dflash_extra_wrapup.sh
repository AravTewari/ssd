#!/usr/bin/env bash
set -euo pipefail

remote_root=/raid/user_data/aravt/ssd
target=/raid/catalyst/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
draft=/raid/catalyst/models/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/5adfac36a234741b344cd906ca3fc6a94d7d5955
predictor=/raid/user_data/aravt/ssd/artifacts/dummy_dflash_predictor
meta=/raid/user_data/aravt/ssd/artifacts/ddtree_prompt_split/training_metadata.json
root=/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/dflash/extra/summary_artifacts
out=/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/dflash/extra/summary.json

port=47000

for batch in 1 2 4; do
  for mode in dflash_ssd_exact_off_normal dflash_ssd_exact_on_oracle; do
    adir="$root/matrix_a/${mode}_b${batch}"
    if ssh catalyst "[ -f '$adir/result.json' ]"; then
      echo "SKIP matrix_a mode=$mode b=$batch"
    else
      echo "RUN matrix_a mode=$mode b=$batch port=$port"
      ssh catalyst "cd $remote_root && export PATH=$remote_root/.venv/bin:/usr/local/bin:/usr/bin:/bin SSD_HF_CACHE=/raid/catalyst/models/hub SSD_DATASET_DIR=/raid/user_data/aravt/datasets CUDA_VISIBLE_DEVICES=4,5 SSD_DIST_PORT=$port; ./.venv/bin/python -O scripts/eval_dflash_predictor.py --target '$target' --draft '$draft' --predictor '$predictor' --training-metadata '$meta' --output-len 128 --gpus 2 --max-prompts 78 --topk 10 --fanout-template-name baseline48 --mode '$mode' --batch-size '$batch' --artifact-dir '$adir'"
    fi
    port=$((port + 1))
  done
done

ssh catalyst "cd $remote_root && ./.venv/bin/python - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, '/raid/user_data/aravt/ssd')
from scripts import eval_dflash_predictor as mod

root = Path('/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/dflash/extra/summary_artifacts')
out = Path('/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/dflash/extra/summary.json')
batch_sizes = [1, 2, 4]
matrix_a_modes = ['dflash_ssd_exact_off_normal', 'dflash_ssd_exact_on_oracle']

matrix_a_results = []
for batch in batch_sizes:
    for mode in matrix_a_modes:
        path = root / 'matrix_a' / f'{mode}_b{batch}' / 'result.json'
        if not path.exists():
            raise SystemExit(f'missing {path}')
        matrix_a_results.append(json.loads(path.read_text()))

recommendation = mod._build_recommendation(matrix_a_results, [])
combined_table = mod._build_combined_markdown_table(matrix_a_results, [])
summary = {
    'training_metadata': '/raid/user_data/aravt/ssd/artifacts/ddtree_prompt_split/training_metadata.json',
    'predictor': '/raid/user_data/aravt/ssd/artifacts/dummy_dflash_predictor',
    'matrix_a': {
        'results': matrix_a_results,
        'quality_metrics': None,
        'recommendation': recommendation,
    },
    'matrix_b': {
        'results': [],
        'best_templates_by_batch': recommendation['decision_rule_c']['best_templates_by_batch'],
    },
    'quality_metrics': None,
    'recommendation': recommendation,
    'combined_markdown_table': combined_table,
}

out.parent.mkdir(parents=True, exist_ok=True)
(root / 'matrix_a_summary.json').write_text(json.dumps(summary['matrix_a'], indent=2, sort_keys=True))
(root / 'matrix_b_summary.json').write_text(json.dumps(summary['matrix_b'], indent=2, sort_keys=True))
out.write_text(json.dumps(summary, indent=2, sort_keys=True))
print(f'WROTE {out}')
PY"
