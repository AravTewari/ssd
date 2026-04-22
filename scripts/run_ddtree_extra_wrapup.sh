#!/usr/bin/env bash
set -euo pipefail

remote_root=/raid/user_data/aravt/ssd
target=/raid/catalyst/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
draft=/raid/catalyst/models/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/5adfac36a234741b344cd906ca3fc6a94d7d5955
predictor=/raid/user_data/aravt/ssd/artifacts/dummy_dflash_predictor
meta=/raid/user_data/aravt/ssd/artifacts/ddtree_prompt_split/training_metadata.json
root=/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/ddtree/extra/summary_artifacts
out=/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/ddtree/extra/summary.json

port=54000
gpu_pair=${GPU_PAIR:-6,7}

for batch in 1 2 4; do
  for tb in 8 16; do
    for fc in 1 2; do
      for mode in ddtree_ssd_exact_off ddtree_ssd_exact_on_oracle; do
        adir="$root/$mode/b${batch}_tb${tb}_fc${fc}"
        if ssh catalyst "[ -f '$adir/result.json' ]"; then
          echo "SKIP mode=$mode b=$batch tb=$tb fc=$fc"
        else
          echo "RUN mode=$mode b=$batch tb=$tb fc=$fc port=$port"
          ssh catalyst "cd $remote_root && export PATH=$remote_root/.venv/bin:/usr/local/bin:/usr/bin:/bin SSD_HF_CACHE=/raid/catalyst/models/hub SSD_DATASET_DIR=/raid/user_data/aravt/datasets CUDA_VISIBLE_DEVICES=$gpu_pair SSD_DIST_PORT=$port; ./.venv/bin/python -O scripts/eval_ddtree.py --target '$target' --draft '$draft' --training-metadata '$meta' --output-len 128 --gpus 2 --max-prompts 78 --mode '$mode' --batch-size '$batch' --tree-budget '$tb' --frontier-count '$fc' --predictor '$predictor' --artifact-dir '$adir'"
        fi
        port=$((port + 1))
      done
    done
  done
done

ssh catalyst "cd $remote_root && ./.venv/bin/python - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, '/raid/user_data/aravt/ssd')
from scripts import eval_ddtree as mod

root = Path('/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/ddtree/extra/summary_artifacts')
out = Path('/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/ddtree/extra/summary.json')
batch_sizes = [1, 2, 4]
tree_budgets = [8, 16]
frontier_counts = [1, 2]
modes = ['ddtree_ssd_exact_off', 'ddtree_ssd_exact_on_oracle']

results = []
for batch in batch_sizes:
    for tb in tree_budgets:
        for fc in frontier_counts:
            for mode in modes:
                path = root / mode / f'b{batch}_tb{tb}_fc{fc}' / 'result.json'
                if not path.exists():
                    raise SystemExit(f'missing {path}')
                results.append(json.loads(path.read_text()))

recommendation = mod._build_recommendation(results)
markdown = mod._format_markdown_table(results)
summary = {
    'results': results,
    'recommendation': recommendation,
    'markdown_table': markdown,
}

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(summary, indent=2, sort_keys=True))
(root / 'summary.md').write_text(markdown + '\n\n```json\n' + json.dumps(recommendation, indent=2, sort_keys=True) + '\n```\n')
print(f'WROTE {out}')
PY"
