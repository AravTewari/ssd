"""Run the basic AR throughput benchmark on Modal (Qwen3-8B, GSM8K, TP=2).

Usage:
    modal run modal_bench/ar_bench.py
"""
from pathlib import Path

import modal

SSD_ROOT = Path(__file__).resolve().parents[1]
DATASET_SRC = Path("/sgl-workspace/dgm/processed_datasets/gsm8k/gsm8k_data_10000.jsonl")

app = modal.App("ssd-ar-bench")

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


@app.function(
    gpu="H200:2",
    image=image,
    volumes={"/cache/hf": hf_vol},
    timeout=60 * 30,
)
def run_ar_bench():
    import os
    import subprocess
    from pathlib import Path

    cache_dir = Path("/cache/hf")
    cache_dir.mkdir(parents=True, exist_ok=True)

    qwen_dir = cache_dir / "models--Qwen--Qwen3-8B"
    has_snapshot = qwen_dir.exists() and any(
        (qwen_dir / "snapshots").glob("*/config.json")
    )
    if not has_snapshot:
        print("[init] downloading Qwen3-8B to volume...")
        from huggingface_hub import snapshot_download

        snapshot_download("Qwen/Qwen3-8B", cache_dir=str(cache_dir))
        hf_vol.commit()
    else:
        print(f"[init] Qwen3-8B already cached at {qwen_dir}")

    cmd = [
        "python",
        "/root/ssd/bench/bench.py",
        "--qwen",
        "--size", "8",
        "--gpus", "2",
        "--b", "1",
        "--numseqs", "32",
        "--input_len", "64",
        "--output_len", "128",
        "--temp", "0.0",
    ]
    print(f"[bench] running: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/ssd" + ":" + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/root/ssd/bench")
    print("=" * 60)
    print("STDOUT")
    print("=" * 60)
    print(proc.stdout)
    if proc.returncode != 0:
        print("=" * 60)
        print("STDERR")
        print("=" * 60)
        print(proc.stderr)
        raise RuntimeError(f"bench.py exited with code {proc.returncode}")

    key_lines = [
        l for l in proc.stdout.splitlines()
        if any(k in l for k in ("Final Prefill", "Final Decode", "Total Throughput", "Best b="))
    ]
    print("=" * 60)
    print("KEY METRICS")
    print("=" * 60)
    for l in key_lines:
        print(l)
    return "\n".join(key_lines)


@app.local_entrypoint()
def main():
    summary = run_ar_bench.remote()
    print("\n===== Bench complete =====")
    print(summary)
