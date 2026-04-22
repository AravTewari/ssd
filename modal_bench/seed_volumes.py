"""Seed shared Modal volumes with the SSD models + datasets so teammates
can mount them directly without re-downloading.

Volumes created / populated:

  ssd-hf-cache  (HF cache layout: models--{org}--{name}/snapshots/{hash}/…)
      - Qwen/Qwen3-0.6B
      - Qwen/Qwen3-1.7B
      - Qwen/Qwen3-4B
      - Qwen/Qwen3-8B
      - z-lab/Qwen3-8B-DFlash-b16  (+ tokenizer symlinks from Qwen3-8B)

  ssd-datasets  (SSD_DATASET_DIR layout)
      - gsm8k/gsm8k_data_10000.jsonl

Usage:
    # seed everything
    modal run modal_bench/seed_volumes.py

    # or just one step
    modal run modal_bench/seed_volumes.py::download_models
    modal run modal_bench/seed_volumes.py::upload_datasets

Once seeded, teammate's Modal function just mounts the two volumes:

    hf_vol = modal.Volume.from_name("ssd-hf-cache")
    ds_vol = modal.Volume.from_name("ssd-datasets")
    @app.function(
        volumes={"/cache/hf": hf_vol, "/cache/datasets": ds_vol},
        ...
    )

and sets `SSD_HF_CACHE=/cache/hf`, `SSD_DATASET_DIR=/cache/datasets`.
"""
from pathlib import Path

import modal

SSD_ROOT = Path(__file__).resolve().parents[1]
DATASET_SRC = Path("/sgl-workspace/dgm/processed_datasets/gsm8k/gsm8k_data_10000.jsonl")

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "z-lab/Qwen3-8B-DFlash-b16",
]

app = modal.App("ssd-seed-volumes")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("hf_transfer", "huggingface_hub[cli]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(
        str(DATASET_SRC),
        "/root/gsm8k_data_10000.jsonl",
        copy=True,
    )
)

hf_vol = modal.Volume.from_name("ssd-hf-cache", create_if_missing=True)
ds_vol = modal.Volume.from_name("ssd-datasets", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/cache/hf": hf_vol},
    timeout=60 * 60,
)
def download_models():
    from huggingface_hub import snapshot_download

    cache_dir = Path("/cache/hf")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for repo in MODELS:
        org, name = repo.split("/", 1)
        target_dir = cache_dir / f"models--{org}--{name}"
        already = target_dir.exists() and any(
            (target_dir / "snapshots").glob("*/config.json")
        )
        if already:
            print(f"[skip] {repo} already present at {target_dir}")
            continue
        print(f"[download] {repo} -> {target_dir}")
        snapshot_download(repo, cache_dir=str(cache_dir))

    # DFlash checkpoint ships no tokenizer — symlink Qwen3-8B's so downstream
    # code (bench.py dflash backend) can load it.
    dflash_base = cache_dir / "models--z-lab--Qwen3-8B-DFlash-b16" / "snapshots"
    qwen8_base = cache_dir / "models--Qwen--Qwen3-8B" / "snapshots"
    if dflash_base.exists() and qwen8_base.exists():
        qwen8_snap = next(qwen8_base.iterdir(), None)
        for dflash_snap in dflash_base.iterdir():
            for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
                src = qwen8_snap / fn if qwen8_snap else None
                dst = dflash_snap / fn
                if src and src.exists() and not dst.exists():
                    dst.symlink_to(src)
                    print(f"[symlink] {dst} -> {src}")

    hf_vol.commit()
    print("[done] models committed to volume ssd-hf-cache")


@app.function(
    image=image,
    volumes={"/cache/datasets": ds_vol},
    timeout=60 * 5,
)
def upload_datasets():
    import shutil

    target_dir = Path("/cache/datasets/gsm8k")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "gsm8k_data_10000.jsonl"
    src = Path("/root/gsm8k_data_10000.jsonl")
    if target.exists() and target.stat().st_size == src.stat().st_size:
        print(f"[skip] {target} already present ({target.stat().st_size} bytes)")
    else:
        shutil.copyfile(src, target)
        print(f"[write] {target} ({target.stat().st_size} bytes)")
    ds_vol.commit()
    print("[done] datasets committed to volume ssd-datasets")


@app.local_entrypoint()
def main():
    print(">>> seeding ssd-datasets")
    upload_datasets.remote()
    print(">>> seeding ssd-hf-cache")
    download_models.remote()
    print("\nAll volumes seeded. Teammate snippet:\n")
    print(
        """
    hf_vol = modal.Volume.from_name("ssd-hf-cache")
    ds_vol = modal.Volume.from_name("ssd-datasets")

    @app.function(
        gpu="H200:2",
        image=image,
        volumes={"/cache/hf": hf_vol, "/cache/datasets": ds_vol},
        ...
    )
        """.strip()
    )
    print(
        "\n    env: SSD_HF_CACHE=/cache/hf  SSD_DATASET_DIR=/cache/datasets"
    )
