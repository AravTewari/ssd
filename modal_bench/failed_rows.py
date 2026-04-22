"""Run the 5 rows that failed locally (SD dflash + 3x SSD ar + SSD dflash) on
Modal H100:2. Reuses the seeded ssd-hf-cache + ssd-datasets volumes.

Usage:
    modal run modal_bench/failed_rows.py --row 5      # SD dflash
    modal run modal_bench/failed_rows.py --row 6      # SSD ar, draft=0.6B
    modal run modal_bench/failed_rows.py --row 7      # SSD ar, draft=1.7B
    modal run modal_bench/failed_rows.py --row 8      # SSD ar, draft=4B
    modal run modal_bench/failed_rows.py --row 9      # SSD dflash
    modal run modal_bench/failed_rows.py --row all    # run 5..9 sequentially
"""
from pathlib import Path

import modal

SSD_ROOT = Path(__file__).resolve().parents[1]
DATASET_SRC = Path("/sgl-workspace/dgm/processed_datasets/gsm8k/gsm8k_data_10000.jsonl")

app = modal.App("ssd-failed-rows")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential", "ninja-build", "libnuma1")
    .pip_install("hf_transfer", "huggingface_hub[cli]")
    .run_commands("pip install -U pip setuptools wheel")
    .add_local_dir(
        str(SSD_ROOT),
        "/root/ssd",
        copy=True,
        ignore=[
            "__pycache__",
            "*.pyc",
            "ssd.egg-info",
            ".git",
            "exp_results*",
            "modal_bench",
            "assets",
        ],
    )
    .run_commands(
        "cd /root/ssd && pip install -e . --no-build-isolation",
    )
    .add_local_file(
        str(DATASET_SRC),
        "/root/datasets/gsm8k/gsm8k_data_10000.jsonl",
        copy=True,
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SSD_HF_CACHE": "/cache/hf",
            "SSD_DATASET_DIR": "/root/datasets",
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        }
    )
)

hf_vol = modal.Volume.from_name("ssd-hf-cache", create_if_missing=True)

# snapshot dirs live under /cache/hf/models--{org}--{name}/snapshots/<hash>/
def _snap(model_dir: str) -> str:
    import os
    base = f"/cache/hf/{model_dir}/snapshots"
    for h in sorted(os.listdir(base)):
        p = os.path.join(base, h)
        if os.path.exists(os.path.join(p, "config.json")):
            return p
    raise FileNotFoundError(f"no snapshot under {base}")


ROWS = {
    5: {"label": "SD_dflash",   "args": ["--qwen","--size","8","--gpus","2","--b","1","--numseqs","32","--input_len","64","--output_len","128","--temp","0.0","--spec","--draft-backend","dflash","--k","15"]},
    6: {"label": "SSD_ar_0.6B", "args": ["--qwen","--size","8","--gpus","2","--b","1","--numseqs","32","--input_len","64","--output_len","128","--temp","0.0","--spec","--async","--k","4","--f","3","--draft","0.6"]},
    7: {"label": "SSD_ar_1.7B", "args": ["--qwen","--size","8","--gpus","2","--b","1","--numseqs","32","--input_len","64","--output_len","128","--temp","0.0","--spec","--async","--k","4","--f","3","--draft","__QWEN_1_7B__"]},
    8: {"label": "SSD_ar_4B",   "args": ["--qwen","--size","8","--gpus","2","--b","1","--numseqs","32","--input_len","64","--output_len","128","--temp","0.0","--spec","--async","--k","4","--f","3","--draft","__QWEN_4B__"]},
    9: {"label": "SSD_dflash",  "args": ["--qwen","--size","8","--gpus","2","--b","1","--numseqs","32","--input_len","64","--output_len","128","--temp","0.0","--spec","--async","--draft-backend","dflash_ssd","--k","15","--f","3"]},
}


@app.function(
    gpu="H100:2",
    image=image,
    volumes={"/cache/hf": hf_vol},
    timeout=60 * 45,
)
def run_row(row: int) -> str:
    import os
    import subprocess

    if row not in ROWS:
        raise ValueError(f"unknown row {row}")

    cfg = ROWS[row]
    args = list(cfg["args"])
    # resolve absolute snapshot paths for 1.7B / 4B drafts
    for i, v in enumerate(args):
        if v == "__QWEN_1_7B__":
            args[i] = _snap("models--Qwen--Qwen3-1.7B")
        elif v == "__QWEN_4B__":
            args[i] = _snap("models--Qwen--Qwen3-4B")

    cmd = ["python", "-u", "/root/ssd/bench/bench.py"] + args
    print(f"[row {row} / {cfg['label']}] running: {' '.join(cmd)}", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ssd" + ":" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    # stream stdout/stderr line-by-line so Modal logs show progress (and the
    # child never blocks on a filled pipe); keep a copy for key-line extraction.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env, cwd="/root/ssd/bench",
    )
    all_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        all_lines.append(line.rstrip("\n"))
    rc = proc.wait()

    key = [l for l in all_lines
           if any(k in l for k in ("Final Prefill","Final Decode","Total Throughput","Best b=","Accept","avg tok","target_full","verify"))]
    tag = f"row {row} / {cfg['label']}"
    if rc != 0:
        return f"[{tag}] FAILED (exit {rc})\n" + "\n".join(key[-20:])
    return f"[{tag}] OK\n" + "\n".join(key[-20:])


@app.local_entrypoint()
def main(row: str = "all"):
    if row == "all":
        # row 9 (SSD dflash) was "不支持" in the original table — needs a
        # dflash-predictor checkpoint we don't have; skip by default.
        rows = [5, 6, 7, 8]
    else:
        rows = [int(row)]
    results = []
    for r in rows:
        print(f"\n>>> launching row {r}")
        try:
            out = run_row.remote(r)
        except Exception as e:
            out = f"[row {r}] EXCEPTION: {e}"
        print(out)
        results.append(out)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for out in results:
        print(out)
        print("-" * 60)
