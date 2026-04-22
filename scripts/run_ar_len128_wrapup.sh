#!/usr/bin/env bash
set -euo pipefail

remote_root=/raid/user_data/aravt/ssd
target=/raid/catalyst/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
draft=/raid/catalyst/models/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
meta=/raid/user_data/aravt/ssd/artifacts/ddtree_prompt_split/training_metadata.json
root=/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/ar/qwen4b/len128/summary_artifacts
out=/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/ar/qwen4b/len128/summary.json

port=58000

for mode in ar_async_exact_off_normal ar_async_exact_on_oracle; do
  for k in 4 6 8 12 15; do
    for b in 1 2 4; do
      adir="$root/$mode/k${k}_b${b}"
      if ssh catalyst "[ -f '$adir/result.json' ]"; then
        echo "SKIP mode=$mode k=$k b=$b"
        port=$((port + 1))
      else
        attempt_port=$port
        ok=0
        for attempt in 1 2 3; do
          echo "RUN mode=$mode k=$k b=$b attempt=$attempt port=$attempt_port"
          if ssh catalyst "cd $remote_root && export PATH=$remote_root/.venv/bin:/usr/local/bin:/usr/bin:/bin SSD_HF_CACHE=/raid/catalyst/models/hub SSD_DATASET_DIR=/raid/user_data/aravt/datasets CUDA_VISIBLE_DEVICES=2,3 SSD_DIST_PORT=$attempt_port; ./.venv/bin/python -O scripts/eval_ar_ssd_baseline.py --target '$target' --draft '$draft' --training-metadata '$meta' --output-len 128 --gpus 2 --max-prompts 78 --fanout-template-name baseline48 --mode '$mode' --batch-size '$b' --k '$k' --artifact-dir '$adir'"; then
            ok=1
            break
          fi
          echo "RETRY mode=$mode k=$k b=$b failed attempt=$attempt"
          ssh catalyst "pkill -f '$adir' || true" || true
          attempt_port=$((attempt_port + 100))
          sleep 5
        done
        if [[ $ok -ne 1 ]]; then
          exit 1
        fi
        port=$((attempt_port + 1))
      fi
    done
  done
done

ssh catalyst "python3 - <<'PY'
import json
from pathlib import Path

root = Path('/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/ar/qwen4b/len128/summary_artifacts')
out = Path('/raid/user_data/aravt/ssd/artifacts/final_wrapup_parallel_20260420/runs/ar/qwen4b/len128/summary.json')
modes = ['ar_async_exact_off_normal', 'ar_async_exact_on_oracle']
k_values = [4, 6, 8, 12, 15]
batch_sizes = [1, 2, 4]
results = []

for mode in modes:
    for k in k_values:
        for b in batch_sizes:
            path = root / mode / f'k{k}_b{b}' / 'result.json'
            if not path.exists():
                raise SystemExit(f'missing {path}')
            results.append(json.loads(path.read_text()))

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    'prompt_count': 78,
    'batch_sizes': batch_sizes,
    'k_values': k_values,
    'modes': modes,
    'results': results,
}, indent=2, sort_keys=True))
print(f'WROTE {out}')
PY"
