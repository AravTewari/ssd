from __future__ import annotations

import argparse
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "final_wrapup_parallel_20260420"
REMOTE_ROOT = Path("/root/ssd")
REMOTE_ARTIFACT_ROOT = REMOTE_ROOT / "artifacts" / "final_wrapup_parallel_20260420"
FIGURE_NAMES = [
    "figure_oracle_ceiling.png",
    "figure_normalized_speedup.png",
    "figure_budget_frontier.png",
    "figure_dflash_branch_cache_failure.png",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("matplotlib", "numpy")
    .add_local_dir(str(ROOT / "scripts"), remote_path=str(REMOTE_ROOT / "scripts"))
    .add_local_dir(str(ARTIFACT_ROOT), remote_path=str(REMOTE_ARTIFACT_ROOT))
)

app = modal.App("ssd-final-wrapup-figures")


@app.function(image=image, gpu="H100:2", timeout=60 * 60)
def render(ar_label: str = "qwen4b", bootstrap_samples: int = 5000) -> dict[str, bytes]:
    import subprocess

    cmd = [
        "python",
        str(REMOTE_ROOT / "scripts" / "render_final_wrapup_figures.py"),
        "--artifact-root",
        str(REMOTE_ARTIFACT_ROOT),
        "--ar-label",
        ar_label,
        "--bootstrap-samples",
        str(bootstrap_samples),
    ]
    subprocess.run(cmd, check=True)
    return {
        name: (REMOTE_ARTIFACT_ROOT / name).read_bytes()
        for name in FIGURE_NAMES
    }


@app.local_entrypoint()
def main(
    ar_label: str = "qwen4b",
    bootstrap_samples: int = 5000,
    output_dir: str = str(ROOT),
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    blobs = render.remote(ar_label=ar_label, bootstrap_samples=bootstrap_samples)
    for name, data in blobs.items():
        (output_path / name).write_bytes(data)
        print(output_path / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final wrap-up figure render on Modal")
    parser.add_argument("--ar-label", default="qwen4b")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--output-dir", default=str(ROOT))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main.local(ar_label=args.ar_label, bootstrap_samples=args.bootstrap_samples, output_dir=args.output_dir)
