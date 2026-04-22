"""Run SSD bench commands locally (non-Modal), with optional auto GPU picking.

Usage (from repo root or bench/):
    python bench/modal_bench.py
    python bench/modal_bench.py --cmd sd_qwen32b
    python bench/modal_bench.py --cmd ssd_qwen0.6b_b8 --extra-args "--numseqs 32"
    python bench/modal_bench.py --cmd ssd_qwen0.6b_b4 --gpus 6,7
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


CUDA_ARCH = "9.0"  # H100

BENCH_COMMANDS: dict[str, tuple[int, str]] = {
    # Qwen 32B (4-5 GPUs)
    "ar_qwen32b": (
        4,
        "python -O bench.py --qwen --size 32 --gpus 4 --b 1 --temp 0 --numseqs 16 --output_len 512",
    ),
    "sd_qwen32b": (
        4,
        "python -O bench.py --qwen --size 32 --gpus 4 --spec --k 6 --b 1 --temp 0 --numseqs 16 --output_len 512",
    ),
    "ssd_qwen32b": (
        5,
        "python -O bench.py --qwen --size 32 --gpus 5 --spec --async --k 7 --f 3 --b 1 --temp 0 --numseqs 16 --output_len 512",
    ),
    # Llama 70B (4-5 GPUs)
    "ar_llama70b": (
        4,
        "python -O bench.py --llama --size 70 --gpus 4 --b 1 --temp 0 --numseqs 16 --output_len 512",
    ),
    "sd_llama70b": (
        4,
        "python -O bench.py --llama --size 70 --gpus 4 --spec --k 6 --b 1 --temp 0 --numseqs 16 --output_len 512",
    ),
    "ssd_llama70b": (
        5,
        "python -O bench.py --llama --size 70 --gpus 5 --spec --async --k 7 --f 3 --b 1 --temp 0 --numseqs 16 --output_len 512",
    ),
    # Qwen 1.7B b=1 (1-2 GPUs)
    "sd_qwen1.7b": (
        1,
        "python -O bench.py --qwen --size 1.7 --gpus 1 --spec --k 6 --b 1 --temp 0 --numseqs 16 --output_len 512",
    ),
    "ssd_qwen1.7b": (
        2,
        "python -O bench.py --qwen --size 1.7 --gpus 2 --spec --async --k 7 --f 3 --b 1 --temp 0 --numseqs 16 --output_len 512",
    ),
    # Qwen 4B b=1 (1-2 GPUs)
    "sd_qwen4b": (
        1,
        "python -O bench.py --qwen --size 4 --gpus 1 --spec --k 6 --b 1 --temp 0 --numseqs 16 --output_len 512",
    ),
    "ssd_qwen4b": (
        2,
        "python -O bench.py --qwen --size 4 --gpus 2 --spec --async --k 7 --f 3 --b 1 --temp 0 --numseqs 16 --output_len 512",
    ),
    # Qwen 0.6B b=1/2/4/8/16/32 (1-2 GPUs)
    "sd_qwen0.6b": (
        1,
        "python -O bench.py --qwen --size 0.6 --gpus 1 --spec --k 6 --b 1 --temp 0 --numseqs 16 --output_len 512",
    ),
    "ssd_qwen0.6b_b2": (
        2,
        "python -O bench.py --qwen --size 0.6 --gpus 2 --spec --async --k 7 --f 3 --b 2 --temp 0 --numseqs 16 --output_len 512",
    ),
    "ssd_qwen0.6b_b4": (
        2,
        "python -O bench.py --qwen --size 0.6 --gpus 2 --spec --async --k 7 --f 3 --b 4 --temp 0 --numseqs 16 --output_len 512",
    ),
    "ssd_qwen0.6b_b8": (
        2,
        "python -O bench.py --qwen --size 0.6 --gpus 2 --spec --async --k 7 --f 3 --b 8 --temp 0 --numseqs 16 --output_len 512",
    ),
    "ssd_qwen0.6b_b16": (
        2,
        "python -O bench.py --qwen --size 0.6 --gpus 2 --spec --async --k 7 --f 3 --b 16 --temp 0 --numseqs 16 --output_len 512",
    ),
    "ssd_qwen0.6b_b32": (
        2,
        "python -O bench.py --qwen --size 0.6 --gpus 2 --spec --async --k 7 --f 3 --b 32 --temp 0 --numseqs 16 --output_len 512",
    ),
}


def _parse_gpu_list(value: str) -> list[int]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("empty GPU list")
    return [int(x) for x in items]


def _query_gpu_rows() -> list[dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    rows: list[dict[str, int]] = []
    for line in out.strip().splitlines():
        idx, util, mem_used, mem_total = [int(x.strip()) for x in line.split(",")]
        rows.append({"index": idx, "util": util, "mem_used": mem_used, "mem_total": mem_total})
    return rows


def _pick_idle_gpus(required: int) -> list[int]:
    rows = _query_gpu_rows()
    idle = [r for r in rows if r["util"] == 0]
    # Prefer truly free cards first.
    idle.sort(key=lambda r: (r["mem_used"], r["index"]))
    if len(idle) < required:
        summary = ", ".join(
            f"GPU{r['index']}: util={r['util']}%, mem={r['mem_used']}MiB/{r['mem_total']}MiB" for r in rows
        )
        raise RuntimeError(
            f"Need {required} GPUs but only found {len(idle)} with utilization=0. "
            f"Current GPUs: {summary}"
        )
    return [r["index"] for r in idle[:required]]


def _build_env(repo_root: Path, visible_gpus: list[int] | None) -> dict[str, str]:
    env = os.environ.copy()
    env["SSD_CUDA_ARCH"] = env.get("SSD_CUDA_ARCH", CUDA_ARCH)
    env["TORCH_CUDA_ARCH_LIST"] = env.get("TORCH_CUDA_ARCH_LIST", CUDA_ARCH)
    env["PYTHONPATH"] = f"{repo_root}:{repo_root / 'bench'}:{env.get('PYTHONPATH', '')}"
    if visible_gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in visible_gpus)

    # If script is launched outside the venv, prefer repo .venv automatically.
    venv_dir = repo_root / ".venv"
    if "VIRTUAL_ENV" not in env and venv_dir.exists():
        env["VIRTUAL_ENV"] = str(venv_dir)
        env["PATH"] = f"{venv_dir / 'bin'}:{env.get('PATH', '')}"

    # Local defaults for required SSD paths (can still be overridden by user env).
    default_hf = Path("/root/.cache/huggingface/hub")
    default_ds = Path("/sgl-workspace/dgm/processed_datasets")
    if "SSD_HF_CACHE" not in env and default_hf.exists():
        env["SSD_HF_CACHE"] = str(default_hf)
    if "SSD_DATASET_DIR" not in env and default_ds.exists():
        env["SSD_DATASET_DIR"] = str(default_ds)
    return env


def _build_cmd(cmd_name: str, extra_args: str, python_exe: str) -> tuple[int, list[str], str]:
    gpu_count, base = BENCH_COMMANDS[cmd_name]
    full = base + (f" {extra_args}" if extra_args else "")
    tokens = shlex.split(full)
    if tokens and tokens[0] == "python":
        tokens[0] = python_exe
    return gpu_count, tokens, full


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SSD benchmarks locally (non-Modal).")
    parser.add_argument("--cmd", default="ar_qwen32b", choices=sorted(BENCH_COMMANDS.keys()))
    parser.add_argument("--extra-args", default="", help="Extra flags appended verbatim.")
    parser.add_argument("--gpus", default="", help="Comma-separated GPU indices, e.g. 6,7")
    parser.add_argument("--no-auto-gpu", action="store_true", help="Do not auto-pick idle GPUs.")
    parser.add_argument("--list", action="store_true", help="List command names and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print command/env and exit.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    args = parser.parse_args()

    if args.list:
        print("Available --cmd values:")
        for name in sorted(BENCH_COMMANDS):
            req, _ = BENCH_COMMANDS[name]
            print(f"  {name:<18}  (needs {req} GPU{'s' if req != 1 else ''})")
        return

    repo_root = Path(__file__).resolve().parents[1]
    bench_dir = repo_root / "bench"
    required_gpus, cmd_tokens, full_cmd = _build_cmd(args.cmd, args.extra_args, args.python)

    if args.gpus:
        visible = _parse_gpu_list(args.gpus)
        if len(visible) < required_gpus:
            raise RuntimeError(
                f"--cmd {args.cmd} needs {required_gpus} GPUs, but --gpus provided only {len(visible)}: {visible}"
            )
    elif args.no_auto_gpu:
        visible = None
    else:
        visible = _pick_idle_gpus(required_gpus)

    env = _build_env(repo_root, visible)

    print("\n" + "=" * 70)
    print(f"SSD Local Runner | cmd: {args.cmd} | GPUs needed: {required_gpus}")
    print(full_cmd)
    if visible is not None:
        print(f"CUDA_VISIBLE_DEVICES={','.join(str(i) for i in visible)}")
    print("=" * 70 + "\n")

    if args.dry_run:
        return

    result = subprocess.run(cmd_tokens, cwd=str(bench_dir), env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
