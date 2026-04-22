import argparse
import json
import math
import os
import random
import struct
import subprocess
import sys
import textwrap
import zlib
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


AR_MODES = [
    "ar_async_exact_off_normal",
    "ar_async_exact_on_oracle",
    "ar_async_normal",
]
DFLASH_MAIN_MATRIX_A_MODES = [
    "exact_dflash",
    "dflash_ssd_exact_off_normal",
    "dflash_ssd_exact_on_oracle",
    "dflash_ssd_predicted_off_oracle",
    "dflash_ssd_predicted_on_oracle",
    "dflash_ssd_predicted_on_normal",
]
DFLASH_EXTRA_MATRIX_A_MODES = [
    "dflash_ssd_exact_off_normal",
    "dflash_ssd_exact_on_oracle",
]
DDTREE_MAIN_MODES = [
    "ddtree",
    "ddtree_ssd_exact_off",
    "ddtree_ssd_exact_on_oracle",
    "ddtree_ssd_predicted_on_surrogate",
]
DDTREE_EXTRA_MODES = [
    "ddtree_ssd_exact_off",
    "ddtree_ssd_exact_on_oracle",
]
FAMILY_COLORS = {
    "AR": (59, 130, 246),
    "DFLASH": (239, 68, 68),
    "DDTREE": (16, 185, 129),
}


FONT_5X7 = {
    " ": ["00000"] * 7,
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "/": ["00001", "00010", "00100", "01000", "10000", "00000", "00000"],
    "%": ["11001", "11010", "00100", "01000", "10110", "00110", "00000"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the final diffusion-as-SSD wrap-up evaluation bundle")
    parser.add_argument("--target", type=str, default=None, help="Qwen3-8B snapshot directory")
    parser.add_argument("--training-metadata", type=str, default=None, help="Held-out prompt split metadata JSON")
    parser.add_argument("--dflash-draft", type=str, default=None, help="Qwen3-8B-DFlash-b16 snapshot directory")
    parser.add_argument("--dflash-predictor", type=str, default=None, help="DFlash predictor checkpoint directory")
    parser.add_argument(
        "--ar-draft-candidates",
        type=str,
        default=None,
        help="Comma-separated draft candidates in name=path form. Defaults to Qwen3-0.6B and Llama-3.2-1B under SSD_HF_CACHE.",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts/final_wrapup", help="Output artifact directory")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4")
    parser.add_argument("--ar-k-values", type=str, default="4,6,8,12,15")
    parser.add_argument("--ddtree-tree-budgets", type=str, default="8,16")
    parser.add_argument("--ddtree-frontier-counts", type=str, default="1,2")
    parser.add_argument("--main-output-len", type=int, default=32)
    parser.add_argument("--extra-output-len", type=int, default=128)
    parser.add_argument("--max-prompts", type=int, default=78)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--base-dist-port", type=int, default=13000)
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing summary JSONs under output-dir when present")
    return parser.parse_args()


def _parse_csv_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_snapshot_path(base_path: str) -> str:
    candidate = Path(base_path)
    if candidate.is_dir():
        if (candidate / "config.json").exists():
            return str(candidate)
        snapshots_dir = candidate / "snapshots"
        if snapshots_dir.is_dir():
            for child in sorted(snapshots_dir.iterdir()):
                if child.is_dir() and (child / "config.json").exists():
                    return str(child)
        for child in sorted(candidate.iterdir()):
            if child.is_dir() and (child / "config.json").exists():
                return str(child)
    raise FileNotFoundError(f"No snapshot (config.json) found under {base_path}")


def _default_ar_candidate_bases() -> dict[str, str]:
    hf_cache = os.environ.get("SSD_HF_CACHE")
    if not hf_cache:
        return {}
    return {
        "qwen0.6b": os.path.join(hf_cache, "models--Qwen--Qwen3-0.6B"),
        "llama1b": os.path.join(hf_cache, "models--meta-llama--Llama-3.2-1B-Instruct"),
    }


def _parse_ar_draft_candidates(arg_value: str | None) -> tuple[dict[str, str], list[str]]:
    candidates = {}
    missing = []
    raw = {}
    if arg_value:
        for item in arg_value.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise ValueError(f"Invalid --ar-draft-candidates entry: {item}")
            name, path = item.split("=", 1)
            raw[name.strip()] = path.strip()
    else:
        raw = _default_ar_candidate_bases()

    for name, base_path in raw.items():
        try:
            candidates[name] = _resolve_snapshot_path(base_path)
        except FileNotFoundError:
            missing.append(name)
    return candidates, missing


def _run_script(script_name: str, script_args: list[str], summary_path: Path, reuse_existing: bool, env: dict) -> dict:
    if reuse_existing and summary_path.exists():
        return _load_json(summary_path)

    cmd = [sys.executable, "-O", str(Path(__file__).resolve().parent / script_name), *script_args]
    completed = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr, flush=True)
        raise RuntimeError(f"Subprocess failed for {script_name}")
    if not summary_path.exists():
        raise RuntimeError(f"Expected summary output missing: {summary_path}")
    return _load_json(summary_path)


def _load_prompt_group_metrics(result_row: dict) -> dict:
    artifact_dir = result_row.get("artifact_dir")
    if not artifact_dir:
        return {}
    metrics_path = Path(artifact_dir) / "prompt_group_metrics.json"
    if not metrics_path.exists():
        return {}
    return _load_json(metrics_path)


def _attach_ar_artifact_dirs(summary: dict, artifact_root: Path, label: str, output_len: int) -> dict:
    base = artifact_root / "runs" / "ar" / label / f"len{output_len}" / "summary_artifacts"
    for row in summary.get("results", []):
        if row.get("artifact_dir"):
            continue
        row["artifact_dir"] = str(
            base / row["mode"] / f"k{int(row['k'])}_b{int(row['batch_size'])}"
        )
    return summary


def _attach_dflash_artifact_dirs(summary: dict, artifact_root: Path, long_regime: bool) -> dict:
    base = artifact_root / "runs" / "dflash" / ("extra" if long_regime else "main") / "summary_artifacts"
    matrix_a = summary.get("matrix_a", {})
    matrix_b = summary.get("matrix_b", {})
    for row in matrix_a.get("results", []):
        if row.get("artifact_dir"):
            continue
        row["artifact_dir"] = str(base / "matrix_a" / f"{row['mode']}_b{int(row['batch_size'])}")
    for row in matrix_b.get("results", []):
        if row.get("artifact_dir"):
            continue
        row["artifact_dir"] = str(
            base / "matrix_b" / f"{row['fanout_template_name']}_b{int(row['batch_size'])}"
        )
    quality_metrics = summary.get("quality_metrics")
    if isinstance(quality_metrics, dict) and not quality_metrics.get("artifact_dir"):
        quality_metrics["artifact_dir"] = str(base / "quality_metrics")
    return summary


def _attach_ddtree_artifact_dirs(summary: dict, artifact_root: Path, long_regime: bool) -> dict:
    base = artifact_root / "runs" / "ddtree" / ("extra" if long_regime else "main") / "summary_artifacts"
    for row in summary.get("results", []):
        if row.get("artifact_dir"):
            continue
        row["artifact_dir"] = str(
            base
            / row["mode"]
            / f"b{int(row['batch_size'])}_tb{int(row['tree_budget'])}_fc{int(row['frontier_count'])}"
        )
    return summary


def _bootstrap_ci(prompt_group_metrics: dict, metric_name: str, samples: int, seed: int) -> dict:
    values = [metrics[metric_name] for metrics in prompt_group_metrics.values() if metrics.get(metric_name) is not None]
    if not values:
        return {"mean": None, "low": None, "high": None}
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    low_idx = int(0.025 * (len(draws) - 1))
    high_idx = int(0.975 * (len(draws) - 1))
    return {
        "mean": sum(values) / len(values),
        "low": draws[low_idx],
        "high": draws[high_idx],
    }


def _annotate_with_confidence(results: list[dict], samples: int, seed: int) -> list[dict]:
    annotated = []
    for idx, row in enumerate(results):
        prompt_group_metrics = _load_prompt_group_metrics(row)
        row = dict(row)
        row["confidence_intervals"] = {
            "throughput_tok_s": _bootstrap_ci(prompt_group_metrics, "throughput_tok_s", samples, seed + idx * 2),
            "accepted_suffix_mean": _bootstrap_ci(prompt_group_metrics, "accepted_suffix_mean", samples, seed + idx * 2 + 1),
        }
        annotated.append(row)
    return annotated


def _best_row(rows: list[dict], *, score_key: str = "throughput_tok_s") -> dict:
    if not rows:
        raise RuntimeError("No rows available for selection")
    return max(rows, key=lambda row: row[score_key])


def _group_by_batch(rows: list[dict]) -> dict[int, list[dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["batch_size"]), []).append(row)
    return grouped


def _relative_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return (a - b) / max(b, 1e-6)


class Canvas:
    def __init__(self, width: int, height: int, background=(255, 255, 255)):
        self.width = width
        self.height = height
        self.pixels = bytearray(background * (width * height))

    def _index(self, x: int, y: int) -> int:
        return (y * self.width + x) * 3

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = self._index(x, y)
            self.pixels[idx:idx + 3] = bytes(color)

    def fill_rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + w)
        y1 = min(self.height, y + h)
        for yy in range(y0, y1):
            row_idx = self._index(x0, yy)
            row_bytes = bytes(color) * max(0, x1 - x0)
            self.pixels[row_idx:row_idx + len(row_bytes)] = row_bytes

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], thickness: int = 1) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in range(-(thickness // 2), thickness // 2 + 1):
                for oy in range(-(thickness // 2), thickness // 2 + 1):
                    self.set_pixel(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def draw_circle(self, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius:
                    self.set_pixel(x, y, color)

    def draw_text(self, x: int, y: int, text: str, color: tuple[int, int, int], scale: int = 2) -> None:
        cursor = x
        for char in text.upper():
            glyph = FONT_5X7.get(char, FONT_5X7[" "])
            for row_idx, row in enumerate(glyph):
                for col_idx, bit in enumerate(row):
                    if bit == "1":
                        self.fill_rect(
                            cursor + col_idx * scale,
                            y + row_idx * scale,
                            scale,
                            scale,
                            color,
                        )
            cursor += 6 * scale

    def save_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            start = y * stride
            raw.extend(self.pixels[start:start + stride])
        compressed = zlib.compress(bytes(raw), level=9)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return (
                struct.pack("!I", len(data))
                + tag
                + data
                + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        png = bytearray(b"\x89PNG\r\n\x1a\n")
        png.extend(chunk(b"IHDR", struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)))
        png.extend(chunk(b"IDAT", compressed))
        png.extend(chunk(b"IEND", b""))
        path.write_bytes(bytes(png))


def _plot_frame(canvas: Canvas, left: int, top: int, width: int, height: int) -> tuple[int, int, int, int]:
    canvas.draw_line(left, top, left, top + height, (0, 0, 0), thickness=2)
    canvas.draw_line(left, top + height, left + width, top + height, (0, 0, 0), thickness=2)
    return left, top, width, height


def _scale_x(value: float, values: list[float], left: int, width: int) -> int:
    if len(values) == 1:
        return left + width // 2
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return left + width // 2
    return left + int((value - vmin) / (vmax - vmin) * width)


def _scale_y(value: float, vmin: float, vmax: float, top: int, height: int) -> int:
    if math.isclose(vmin, vmax):
        return top + height // 2
    return top + height - int((value - vmin) / (vmax - vmin) * height)


def _draw_legend(canvas: Canvas, x: int, y: int, families: list[str]) -> None:
    for idx, family in enumerate(families):
        yy = y + idx * 20
        color = FAMILY_COLORS[family]
        canvas.fill_rect(x, yy, 12, 12, color)
        canvas.draw_text(x + 18, yy - 1, family, (0, 0, 0), scale=2)


def _render_oracle_ceiling(path: Path, ar_points: list[dict], dflash_points: list[dict], ddtree_points: list[dict]) -> None:
    canvas = Canvas(960, 540)
    left, top, width, height = _plot_frame(canvas, 80, 40, 760, 380)
    batches = sorted({int(row["batch_size"]) for row in [*ar_points, *dflash_points, *ddtree_points]})
    series = {
        "AR": ar_points,
        "DFLASH": dflash_points,
        "DDTREE": ddtree_points,
    }
    y_values = [row["throughput_tok_s"] for rows in series.values() for row in rows]
    y_min = 0.0
    y_max = max(y_values) * 1.1 if y_values else 1.0
    for batch in batches:
        x = _scale_x(batch, batches, left, width)
        canvas.draw_line(x, top + height, x, top + height + 8, (0, 0, 0), thickness=2)
        canvas.draw_text(x - 6, top + height + 14, str(batch), (0, 0, 0), scale=2)
    for family, rows in series.items():
        color = FAMILY_COLORS[family]
        rows = sorted(rows, key=lambda row: row["batch_size"])
        prev = None
        for row in rows:
            x = _scale_x(row["batch_size"], batches, left, width)
            y = _scale_y(row["throughput_tok_s"], y_min, y_max, top, height)
            canvas.draw_circle(x, y, 5, color)
            if prev is not None:
                canvas.draw_line(prev[0], prev[1], x, y, color, thickness=2)
            prev = (x, y)
    _draw_legend(canvas, 700, 440, ["AR", "DFLASH", "DDTREE"])
    canvas.save_png(path)


def _render_normalized_speedup(path: Path, ar_points: list[dict], dflash_points: list[dict], ddtree_points: list[dict]) -> None:
    canvas = Canvas(960, 540)
    left, top, width, height = _plot_frame(canvas, 80, 40, 760, 380)
    batches = sorted({int(row["batch_size"]) for row in [*ar_points, *dflash_points, *ddtree_points]})
    series = {
        "AR": ar_points,
        "DFLASH": dflash_points,
        "DDTREE": ddtree_points,
    }
    y_values = [row["normalized_speedup"] for rows in series.values() for row in rows if row["normalized_speedup"] is not None]
    y_min = min(1.0, min(y_values)) * 0.95 if y_values else 0.9
    y_max = max(y_values) * 1.05 if y_values else 1.1
    baseline_y = _scale_y(1.0, y_min, y_max, top, height)
    canvas.draw_line(left, baseline_y, left + width, baseline_y, (160, 160, 160), thickness=1)
    for batch in batches:
        x = _scale_x(batch, batches, left, width)
        canvas.draw_line(x, top + height, x, top + height + 8, (0, 0, 0), thickness=2)
        canvas.draw_text(x - 6, top + height + 14, str(batch), (0, 0, 0), scale=2)
    for family, rows in series.items():
        color = FAMILY_COLORS[family]
        rows = sorted(rows, key=lambda row: row["batch_size"])
        prev = None
        for row in rows:
            if row["normalized_speedup"] is None:
                continue
            x = _scale_x(row["batch_size"], batches, left, width)
            y = _scale_y(row["normalized_speedup"], y_min, y_max, top, height)
            canvas.draw_circle(x, y, 5, color)
            if prev is not None:
                canvas.draw_line(prev[0], prev[1], x, y, color, thickness=2)
            prev = (x, y)
    _draw_legend(canvas, 700, 440, ["AR", "DFLASH", "DDTREE"])
    canvas.save_png(path)


def _render_budget_frontier(path: Path, ar_points: list[dict], dflash_points: list[dict], ddtree_points: list[dict]) -> None:
    canvas = Canvas(960, 540)
    left, top, width, height = _plot_frame(canvas, 80, 40, 760, 380)
    all_points = [*ar_points, *dflash_points, *ddtree_points]
    xs = [row["accepted_suffix_mean"] for row in all_points if row.get("accepted_suffix_mean") is not None]
    ys = [row["throughput_tok_s"] for row in all_points]
    x_min = min(xs) * 0.95 if xs else 0.0
    x_max = max(xs) * 1.05 if xs else 1.0
    y_min = 0.0
    y_max = max(ys) * 1.1 if ys else 1.0
    series = {
        "AR": ar_points,
        "DFLASH": dflash_points,
        "DDTREE": ddtree_points,
    }
    for family, rows in series.items():
        color = FAMILY_COLORS[family]
        for row in rows:
            if row.get("accepted_suffix_mean") is None:
                continue
            x = left + int((row["accepted_suffix_mean"] - x_min) / max(x_max - x_min, 1e-6) * width)
            y = _scale_y(row["throughput_tok_s"], y_min, y_max, top, height)
            canvas.draw_circle(x, y, 4, color)
    _draw_legend(canvas, 700, 440, ["AR", "DFLASH", "DDTREE"])
    canvas.save_png(path)


def _render_error_bars(path: Path, ar_points: list[dict], dflash_points: list[dict], ddtree_points: list[dict]) -> None:
    canvas = Canvas(960, 640)
    panels = [
        ("throughput_tok_s", ar_points, dflash_points, ddtree_points, 50),
        ("accepted_suffix_mean", ar_points, dflash_points, ddtree_points, 340),
    ]
    for metric_name, ar_rows, dflash_rows, ddtree_rows, top in panels:
        left, _, width, height = _plot_frame(canvas, 80, top, 760, 210)
        batches = sorted({int(row["batch_size"]) for row in [*ar_rows, *dflash_rows, *ddtree_rows]})
        values = []
        for rows in [ar_rows, dflash_rows, ddtree_rows]:
            for row in rows:
                ci = row["confidence_intervals"][metric_name]
                if ci["high"] is not None:
                    values.append(ci["high"])
        y_min = 0.0
        y_max = max(values) * 1.1 if values else 1.0
        for batch in batches:
            x = _scale_x(batch, batches, left, width)
            canvas.draw_line(x, top + height, x, top + height + 8, (0, 0, 0), thickness=2)
            canvas.draw_text(x - 6, top + height + 14, str(batch), (0, 0, 0), scale=2)
        for family, rows, dx in [("AR", ar_rows, -8), ("DFLASH", dflash_rows, 0), ("DDTREE", ddtree_rows, 8)]:
            color = FAMILY_COLORS[family]
            for row in rows:
                ci = row["confidence_intervals"][metric_name]
                if ci["mean"] is None:
                    continue
                x = _scale_x(row["batch_size"], batches, left, width) + dx
                y = _scale_y(ci["mean"], y_min, y_max, top, height)
                y_low = _scale_y(ci["low"], y_min, y_max, top, height)
                y_high = _scale_y(ci["high"], y_min, y_max, top, height)
                canvas.draw_line(x, y_low, x, y_high, color, thickness=1)
                canvas.draw_line(x - 4, y_low, x + 4, y_low, color, thickness=1)
                canvas.draw_line(x - 4, y_high, x + 4, y_high, color, thickness=1)
                canvas.draw_circle(x, y, 4, color)
    _draw_legend(canvas, 700, 580, ["AR", "DFLASH", "DDTREE"])
    canvas.save_png(path)


def _prepare_ar_run(
    args,
    artifact_root: Path,
    label: str,
    draft_path: str,
    output_len: int,
    modes: list[str],
    port_offset: int,
) -> dict:
    summary_path = artifact_root / "runs" / "ar" / label / f"len{output_len}" / "summary.json"
    env = os.environ.copy()
    env["SSD_DIST_PORT"] = str(args.base_dist_port + port_offset)
    summary = _run_script(
        "eval_ar_ssd_baseline.py",
        [
            "--target", args.target,
            "--draft", draft_path,
            "--training-metadata", args.training_metadata,
            "--output-len", str(output_len),
            "--batch-sizes", args.batch_sizes,
            "--max-prompts", str(args.max_prompts),
            "--k-values", args.ar_k_values,
            "--modes", ",".join(modes),
            "--out", str(summary_path),
        ],
        summary_path=summary_path,
        reuse_existing=args.reuse_existing,
        env=env,
    )
    return _attach_ar_artifact_dirs(summary, artifact_root, label, output_len)


def _prepare_dflash_run(args, artifact_root: Path, output_len: int, long_regime: bool, port_offset: int) -> dict:
    summary_path = artifact_root / "runs" / "dflash" / ("extra" if long_regime else "main") / "summary.json"
    env = os.environ.copy()
    env["SSD_DIST_PORT"] = str(args.base_dist_port + port_offset)
    script_args = [
        "--target", args.target,
        "--draft", args.dflash_draft,
        "--predictor", args.dflash_predictor,
        "--training-metadata", args.training_metadata,
        "--output-len", str(output_len),
        "--batch-sizes", args.batch_sizes,
        "--max-prompts", str(args.max_prompts),
        "--out", str(summary_path),
    ]
    if long_regime:
        script_args.extend([
            "--matrix-a-modes", ",".join(DFLASH_EXTRA_MATRIX_A_MODES),
            "--skip-matrix-b",
            "--skip-quality-metrics",
        ])
    summary = _run_script(
        "eval_dflash_predictor.py",
        script_args,
        summary_path=summary_path,
        reuse_existing=args.reuse_existing,
        env=env,
    )
    return _attach_dflash_artifact_dirs(summary, artifact_root, long_regime)


def _prepare_ddtree_run(args, artifact_root: Path, output_len: int, long_regime: bool, port_offset: int) -> dict:
    summary_path = artifact_root / "runs" / "ddtree" / ("extra" if long_regime else "main") / "summary.json"
    env = os.environ.copy()
    env["SSD_DIST_PORT"] = str(args.base_dist_port + port_offset)
    modes = DDTREE_EXTRA_MODES if long_regime else DDTREE_MAIN_MODES
    summary = _run_script(
        "eval_ddtree.py",
        [
            "--target", args.target,
            "--draft", args.dflash_draft,
            "--predictor", args.dflash_predictor,
            "--training-metadata", args.training_metadata,
            "--output-len", str(output_len),
            "--batch-sizes", args.batch_sizes,
            "--tree-budgets", args.ddtree_tree_budgets,
            "--frontier-counts", args.ddtree_frontier_counts,
            "--modes", ",".join(modes),
            "--max-prompts", str(args.max_prompts),
            "--out", str(summary_path),
        ],
        summary_path=summary_path,
        reuse_existing=args.reuse_existing,
        env=env,
    )
    return _attach_ddtree_artifact_dirs(summary, artifact_root, long_regime)


def _extract_best_ar_frontier(ar_results: list[dict], mode: str) -> list[dict]:
    grouped = _group_by_batch([row for row in ar_results if row["mode"] == mode])
    return [_best_row(rows) for _, rows in sorted(grouped.items())]


def _extract_best_ddtree_frontier(ddtree_results: list[dict], mode: str) -> list[dict]:
    grouped = _group_by_batch([row for row in ddtree_results if row["mode"] == mode])
    return [_best_row(rows) for _, rows in sorted(grouped.items())]


def _index_by_batch(rows: list[dict]) -> dict[int, dict]:
    return {int(row["batch_size"]): row for row in rows}


def _annotate_normalized_speedup(oracle_rows: list[dict], exact_off_rows: list[dict]) -> list[dict]:
    off_by_batch = _index_by_batch(exact_off_rows)
    out = []
    for row in oracle_rows:
        base = off_by_batch[int(row["batch_size"])]
        row = dict(row)
        row["normalized_speedup"] = row["throughput_tok_s"] / max(base["throughput_tok_s"], 1e-6)
        out.append(row)
    return out


def _build_main_decomposition_rows(ar_results: list[dict], dflash_results: list[dict], ddtree_results: list[dict]) -> list[dict]:
    rows = []
    for mode_label, ar_mode, dflash_mode, ddtree_mode in [
        ("exact_off", "ar_async_exact_off_normal", "dflash_ssd_exact_off_normal", "ddtree_ssd_exact_off"),
        ("exact_on_oracle", "ar_async_exact_on_oracle", "dflash_ssd_exact_on_oracle", "ddtree_ssd_exact_on_oracle"),
        ("realized", "ar_async_normal", "dflash_ssd_predicted_on_normal", "ddtree_ssd_predicted_on_surrogate"),
    ]:
        ar_frontier = _extract_best_ar_frontier(ar_results, ar_mode)
        ddtree_frontier = _extract_best_ddtree_frontier(ddtree_results, ddtree_mode)
        dflash_rows = sorted(
            [row for row in dflash_results if row["mode"] == dflash_mode],
            key=lambda row: row["batch_size"],
        )
        for family, family_rows in [
            ("AR", ar_frontier),
            ("DFLASH", dflash_rows),
            ("DDTREE", ddtree_frontier),
        ]:
            for row in family_rows:
                rows.append(
                    {
                        "family": family,
                        "mode": mode_label,
                        "batch_size": row["batch_size"],
                        "throughput_tok_s": row["throughput_tok_s"],
                        "accepted_suffix_mean": row.get("accepted_suffix_mean"),
                        "cache_hit_mean": row.get("cache_hit_mean"),
                        "setting": {
                            key: row[key]
                            for key in ["k", "fanout_template_name", "tree_budget", "frontier_count"]
                            if key in row
                        },
                        "confidence_intervals": row["confidence_intervals"],
                    }
                )
    return rows


def _build_appendix_equivalence(dflash_results: list[dict], ddtree_results: list[dict]) -> dict:
    dflash_pairs = []
    exact_by_batch = _index_by_batch([row for row in dflash_results if row["mode"] == "exact_dflash"])
    exact_off_by_batch = _index_by_batch([row for row in dflash_results if row["mode"] == "dflash_ssd_exact_off_normal"])
    for batch_size in sorted(exact_by_batch):
        exact_row = exact_by_batch[batch_size]
        off_row = exact_off_by_batch[batch_size]
        dflash_pairs.append(
            {
                "batch_size": batch_size,
                "standalone_tok_s": exact_row["throughput_tok_s"],
                "exact_off_tok_s": off_row["throughput_tok_s"],
                "delta_frac": _relative_delta(off_row["throughput_tok_s"], exact_row["throughput_tok_s"]),
            }
        )

    ddtree_pairs = []
    standalone_rows = [row for row in ddtree_results if row["mode"] == "ddtree"]
    exact_off_rows = [row for row in ddtree_results if row["mode"] == "ddtree_ssd_exact_off"]
    for off_row in exact_off_rows:
        matching = [
            row for row in standalone_rows
            if int(row["batch_size"]) == int(off_row["batch_size"])
            and int(row["tree_budget"]) == int(off_row["tree_budget"])
            and int(row["frontier_count"]) == int(off_row["frontier_count"])
        ]
        if not matching:
            continue
        standalone = _best_row(matching)
        ddtree_pairs.append(
            {
                "batch_size": int(off_row["batch_size"]),
                "tree_budget": int(off_row["tree_budget"]),
                "frontier_count": int(off_row["frontier_count"]),
                "standalone_tok_s": standalone["throughput_tok_s"],
                "exact_off_tok_s": off_row["throughput_tok_s"],
                "delta_frac": _relative_delta(off_row["throughput_tok_s"], standalone["throughput_tok_s"]),
            }
        )
    return {
        "dflash": dflash_pairs,
        "ddtree": ddtree_pairs,
    }


def _build_extra_regime_rows(ar_results: list[dict], dflash_results: list[dict], ddtree_results: list[dict]) -> list[dict]:
    rows = []
    ar_exact_off = _extract_best_ar_frontier(ar_results, "ar_async_exact_off_normal")
    ar_oracle = _extract_best_ar_frontier(ar_results, "ar_async_exact_on_oracle")
    dflash_exact_off = sorted([row for row in dflash_results if row["mode"] == "dflash_ssd_exact_off_normal"], key=lambda row: row["batch_size"])
    dflash_oracle = sorted([row for row in dflash_results if row["mode"] == "dflash_ssd_exact_on_oracle"], key=lambda row: row["batch_size"])
    ddtree_exact_off = _extract_best_ddtree_frontier(ddtree_results, "ddtree_ssd_exact_off")
    ddtree_oracle = _extract_best_ddtree_frontier(ddtree_results, "ddtree_ssd_exact_on_oracle")
    for family, off_rows, oracle_rows in [
        ("AR", ar_exact_off, ar_oracle),
        ("DFLASH", dflash_exact_off, dflash_oracle),
        ("DDTREE", ddtree_exact_off, ddtree_oracle),
    ]:
        off_by_batch = _index_by_batch(off_rows)
        for oracle_row in oracle_rows:
            base_row = off_by_batch[int(oracle_row["batch_size"])]
            rows.append(
                {
                    "family": family,
                    "batch_size": int(oracle_row["batch_size"]),
                    "exact_off_tok_s": base_row["throughput_tok_s"],
                    "exact_on_oracle_tok_s": oracle_row["throughput_tok_s"],
                    "oracle_vs_exact_off_delta_frac": _relative_delta(
                        oracle_row["throughput_tok_s"],
                        base_row["throughput_tok_s"],
                    ),
                    "accepted_suffix_exact_off": base_row.get("accepted_suffix_mean"),
                    "accepted_suffix_exact_on_oracle": oracle_row.get("accepted_suffix_mean"),
                }
            )
    return rows


def _build_recommendation(
    main_oracle_ar: list[dict],
    main_oracle_dflash: list[dict],
    main_oracle_ddtree: list[dict],
    main_exact_off_dflash: list[dict],
    main_exact_off_ddtree: list[dict],
    extra_rows: list[dict],
) -> dict:
    oracle_comparison = []
    diffusion_beats_ar = False
    for family_name, rows in [("DFLASH", main_oracle_dflash), ("DDTREE", main_oracle_ddtree)]:
        by_batch = _index_by_batch(rows)
        ar_by_batch = _index_by_batch(main_oracle_ar)
        for batch_size, row in sorted(by_batch.items()):
            ar_row = ar_by_batch[batch_size]
            delta = _relative_delta(row["throughput_tok_s"], ar_row["throughput_tok_s"])
            oracle_comparison.append(
                {
                    "family": family_name,
                    "batch_size": batch_size,
                    "oracle_tok_s": row["throughput_tok_s"],
                    "ar_oracle_tok_s": ar_row["throughput_tok_s"],
                    "delta_vs_ar_oracle_frac": delta,
                }
            )
            if delta is not None and delta > 0:
                diffusion_beats_ar = True

    self_speedups = []
    diffusion_self_gain = False
    for family_name, oracle_rows, exact_off_rows in [
        ("DFLASH", main_oracle_dflash, main_exact_off_dflash),
        ("DDTREE", main_oracle_ddtree, main_exact_off_ddtree),
    ]:
        exact_off_by_batch = _index_by_batch(exact_off_rows)
        for row in oracle_rows:
            delta = _relative_delta(
                row["throughput_tok_s"],
                exact_off_by_batch[int(row["batch_size"])]["throughput_tok_s"],
            )
            self_speedups.append(
                {
                    "family": family_name,
                    "regime": "main",
                    "batch_size": int(row["batch_size"]),
                    "oracle_vs_exact_off_delta_frac": delta,
                }
            )
            if delta is not None and delta >= 0.05:
                diffusion_self_gain = True
        for row in extra_rows:
            if row["family"] != family_name:
                continue
            self_speedups.append(
                {
                    "family": family_name,
                    "regime": "extra",
                    "batch_size": row["batch_size"],
                    "oracle_vs_exact_off_delta_frac": row["oracle_vs_exact_off_delta_frac"],
                }
            )
            if row["oracle_vs_exact_off_delta_frac"] is not None and row["oracle_vs_exact_off_delta_frac"] >= 0.05:
                diffusion_self_gain = True

    stop = (not diffusion_beats_ar) and (not diffusion_self_gain)
    return {
        "stop_project": stop,
        "message": (
            "Diffusion never beats AR on oracle throughput and never exceeds its own exact-off baseline by 5%; the negative result is stable."
            if stop
            else "At least one diffusion family still shows meaningful headroom under the final wrap-up criteria."
        ),
        "oracle_vs_ar": oracle_comparison,
        "extra_regime_self_speedups": self_speedups,
    }


def _markdown_table_setup(args, ar_candidates: dict[str, str], missing_ar: list[str]) -> str:
    lines = [
        "| field | value |",
        "| --- | --- |",
        f"| target | `{args.target}` |",
        f"| DFlash/DDTree draft | `{args.dflash_draft}` |",
        f"| DFlash predictor | `{args.dflash_predictor}` |",
        f"| hardware | `2x B200` |",
        f"| held-out prompts | `{args.max_prompts}` prompt groups from `{args.training_metadata}` |",
        f"| main output length | `{args.main_output_len}` |",
        f"| extra output length | `{args.extra_output_len}` |",
        f"| AR draft candidates used | `{', '.join(f'{k}={v}' for k, v in ar_candidates.items())}` |",
        f"| AR draft candidates missing | `{', '.join(missing_ar) if missing_ar else 'none'}` |",
    ]
    return "\n".join(lines)


def _markdown_table_main(rows: list[dict]) -> str:
    lines = [
        "| family | mode | b | tok/s | accepted suffix | cache hit | 95% CI tok/s | 95% CI acc | setting |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        ci_tok = row["confidence_intervals"]["throughput_tok_s"]
        ci_acc = row["confidence_intervals"]["accepted_suffix_mean"]
        setting = ", ".join(f"{k}={v}" for k, v in row["setting"].items()) if row["setting"] else "fixed"
        ci_tok_text = "n/a" if ci_tok["low"] is None else f"[{ci_tok['low']:.2f}, {ci_tok['high']:.2f}]"
        ci_acc_text = "n/a" if ci_acc["low"] is None else f"[{ci_acc['low']:.3f}, {ci_acc['high']:.3f}]"
        lines.append(
            f"| {row['family']} | {row['mode']} | {row['batch_size']} | "
            f"{row['throughput_tok_s']:.2f} | {row['accepted_suffix_mean']:.3f} | "
            f"{row['cache_hit_mean'] if row['cache_hit_mean'] is not None else 'n/a'} | "
            f"{ci_tok_text} | {ci_acc_text} | {setting} |"
        )
    return "\n".join(lines)


def _markdown_table_extra(rows: list[dict]) -> str:
    lines = [
        "| family | b | exact-off tok/s | exact-on-oracle tok/s | oracle delta | exact-off acc | oracle acc |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        delta_pct = "n/a" if row["oracle_vs_exact_off_delta_frac"] is None else f"{row['oracle_vs_exact_off_delta_frac'] * 100:.1f}%"
        lines.append(
            f"| {row['family']} | {row['batch_size']} | {row['exact_off_tok_s']:.2f} | "
            f"{row['exact_on_oracle_tok_s']:.2f} | {delta_pct} | "
            f"{row['accepted_suffix_exact_off']:.3f} | {row['accepted_suffix_exact_on_oracle']:.3f} |"
        )
    return "\n".join(lines)


def _markdown_table_equivalence(data: dict) -> str:
    lines = [
        "### DFlash",
        "| b | standalone tok/s | exact-off tok/s | delta |",
        "| --- | --- | --- | --- |",
    ]
    for row in data["dflash"]:
        delta = "n/a" if row["delta_frac"] is None else f"{row['delta_frac'] * 100:.1f}%"
        lines.append(
            f"| {row['batch_size']} | {row['standalone_tok_s']:.2f} | {row['exact_off_tok_s']:.2f} | {delta} |"
        )
    lines.extend([
        "",
        "### DDTree",
        "| b | N | F | standalone tok/s | exact-off tok/s | delta |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for row in data["ddtree"]:
        delta = "n/a" if row["delta_frac"] is None else f"{row['delta_frac'] * 100:.1f}%"
        lines.append(
            f"| {row['batch_size']} | {row['tree_budget']} | {row['frontier_count']} | "
            f"{row['standalone_tok_s']:.2f} | {row['exact_off_tok_s']:.2f} | {delta} |"
        )
    return "\n".join(lines)


def _load_dream_appendix_note() -> str:
    dream_readme = Path(__file__).resolve().parent.parent / "README_DREAM_RESULTS.md"
    if not dream_readme.exists():
        return "Dream appendix note unavailable: README_DREAM_RESULTS.md not found."
    text = dream_readme.read_text(encoding="utf-8")
    summary = []
    for line in text.splitlines():
        if "Dream diffusion" in line or "AR baseline" in line or "5.9x" in line or "9.2x" in line:
            summary.append(line.strip())
    if not summary:
        summary.append("Dream was kept as an appendix-only exploratory failure mode; see README_DREAM_RESULTS.md.")
    return "\n".join(f"- {line}" for line in summary[:6])


def _generate_artifacts(
    args,
    artifact_root: Path,
    ar_candidates: dict[str, str],
    missing_ar: list[str],
    ar_main_results: list[dict],
    ar_extra_results: list[dict],
    dflash_main_results: list[dict],
    dflash_main_matrix_b: list[dict],
    dflash_extra_results: list[dict],
    ddtree_main_results: list[dict],
    ddtree_extra_results: list[dict],
) -> dict:
    ar_main_results = _annotate_with_confidence(ar_main_results, args.bootstrap_samples, args.bootstrap_seed)
    ar_extra_results = _annotate_with_confidence(ar_extra_results, args.bootstrap_samples, args.bootstrap_seed + 1000)
    dflash_main_results = _annotate_with_confidence(dflash_main_results, args.bootstrap_samples, args.bootstrap_seed + 2000)
    dflash_main_matrix_b = _annotate_with_confidence(dflash_main_matrix_b, args.bootstrap_samples, args.bootstrap_seed + 3000) if dflash_main_matrix_b else []
    dflash_extra_results = _annotate_with_confidence(dflash_extra_results, args.bootstrap_samples, args.bootstrap_seed + 4000)
    ddtree_main_results = _annotate_with_confidence(ddtree_main_results, args.bootstrap_samples, args.bootstrap_seed + 5000)
    ddtree_extra_results = _annotate_with_confidence(ddtree_extra_results, args.bootstrap_samples, args.bootstrap_seed + 6000)

    ar_main_oracle = _annotate_normalized_speedup(
        _extract_best_ar_frontier(ar_main_results, "ar_async_exact_on_oracle"),
        _extract_best_ar_frontier(ar_main_results, "ar_async_exact_off_normal"),
    )
    dflash_main_oracle = _annotate_normalized_speedup(
        sorted([row for row in dflash_main_results if row["mode"] == "dflash_ssd_exact_on_oracle"], key=lambda row: row["batch_size"]),
        sorted([row for row in dflash_main_results if row["mode"] == "dflash_ssd_exact_off_normal"], key=lambda row: row["batch_size"]),
    )
    ddtree_main_oracle = _annotate_normalized_speedup(
        _extract_best_ddtree_frontier(ddtree_main_results, "ddtree_ssd_exact_on_oracle"),
        _extract_best_ddtree_frontier(ddtree_main_results, "ddtree_ssd_exact_off"),
    )

    main_table_rows = _build_main_decomposition_rows(ar_main_results, dflash_main_results, ddtree_main_results)
    appendix_equivalence = _build_appendix_equivalence(dflash_main_results, ddtree_main_results)
    extra_rows = _build_extra_regime_rows(ar_extra_results, dflash_extra_results, ddtree_extra_results)
    recommendation = _build_recommendation(
        ar_main_oracle,
        dflash_main_oracle,
        ddtree_main_oracle,
        sorted([row for row in dflash_main_results if row["mode"] == "dflash_ssd_exact_off_normal"], key=lambda row: row["batch_size"]),
        _extract_best_ddtree_frontier(ddtree_main_results, "ddtree_ssd_exact_off"),
        extra_rows,
    )

    ar_oracle_frontier_points = [row for row in ar_main_results if row["mode"] == "ar_async_exact_on_oracle"]
    ddtree_oracle_frontier_points = [row for row in ddtree_main_results if row["mode"] == "ddtree_ssd_exact_on_oracle"]
    dflash_oracle_frontier_points = [row for row in dflash_main_results if row["mode"] == "dflash_ssd_exact_on_oracle"]

    final_tables = [
        "# Diffusion-as-SSD Final Wrap-Up",
        "",
        "## Table 1: Setup",
        _markdown_table_setup(args, ar_candidates, missing_ar),
        "",
        "## Table 2: Main Decomposition Matrix",
        _markdown_table_main(main_table_rows),
        "",
        "## Figure Captions",
        "- `figure_oracle_ceiling.png`: best oracle throughput by batch size for AR, DFlash, and DDTree.",
        "- `figure_normalized_speedup.png`: oracle throughput normalized by each family's exact-off throughput.",
        "- `figure_budget_frontier.png`: oracle throughput vs accepted suffix over swept AR and DDTree budgets, with DFlash fixed points.",
        "- `figure_error_bars.png`: 95% bootstrap confidence intervals for oracle throughput and accepted suffix.",
        "",
        "## Appendix Table: Standalone vs Exact-Off",
        _markdown_table_equivalence(appendix_equivalence),
        "",
        "## Appendix DFlash Diagnostics",
        "See `appendix_dflash_diagnostics.md` for the full diagnostic matrix and branch-template sweep.",
        "",
        "## Appendix Extra Regime",
        _markdown_table_extra(extra_rows),
        "",
        "## Appendix Dream Note",
        _load_dream_appendix_note(),
        "",
        "## Final Recommendation",
        "```json",
        json.dumps(recommendation, indent=2, sort_keys=True),
        "```",
    ]
    final_tables_text = "\n".join(final_tables) + "\n"

    appendix_dflash_lines = [
        "# DFlash Appendix Diagnostics",
        "",
        "## Main Diagnostic Matrix",
        dflash_main_results[0].get("combined_markdown_table", "See matrix A summary JSON for details.")
        if isinstance(dflash_main_results[0], dict) and "combined_markdown_table" in dflash_main_results[0]
        else "See `matrix_a_summary.json` and `matrix_b_summary.json` under the DFlash run directory.",
        "",
    ]
    if dflash_main_matrix_b:
        appendix_dflash_lines.extend(
            [
                "## Branch-Template Sweep",
                "| template | b | tok/s | cache hit | cache-committed frac | joint recall |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in dflash_main_matrix_b:
            appendix_dflash_lines.append(
                f"| {row['fanout_template_name']} | {row['batch_size']} | {row['throughput_tok_s']:.2f} | "
                f"{row['cache_hit_mean'] if row['cache_hit_mean'] is not None else 'n/a'} | "
                f"{row['cache_committed_token_fraction'] if row['cache_committed_token_fraction'] is not None else 'n/a'} | "
                f"{row['joint_branch_recall'] if row['joint_branch_recall'] is not None else 'n/a'} |"
            )
    appendix_dflash_text = "\n".join(appendix_dflash_lines) + "\n"

    appendix_extra_text = "\n".join(
        [
            "# Extra SSD-Favorable Regime",
            "",
            _markdown_table_extra(extra_rows),
            "",
            "```json",
            json.dumps(recommendation, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )

    _write_text(artifact_root / "final_tables.md", final_tables_text)
    _write_text(artifact_root / "appendix_dflash_diagnostics.md", appendix_dflash_text)
    _write_text(artifact_root / "appendix_extra_regime.md", appendix_extra_text)

    _render_oracle_ceiling(artifact_root / "figure_oracle_ceiling.png", ar_main_oracle, dflash_main_oracle, ddtree_main_oracle)
    _render_normalized_speedup(artifact_root / "figure_normalized_speedup.png", ar_main_oracle, dflash_main_oracle, ddtree_main_oracle)
    _render_budget_frontier(artifact_root / "figure_budget_frontier.png", ar_oracle_frontier_points, dflash_oracle_frontier_points, ddtree_oracle_frontier_points)
    _render_error_bars(artifact_root / "figure_error_bars.png", ar_main_oracle, dflash_main_oracle, ddtree_main_oracle)

    summary = {
        "setup": {
            "target": args.target,
            "dflash_draft": args.dflash_draft,
            "dflash_predictor": args.dflash_predictor,
            "ar_candidates": ar_candidates,
            "missing_ar_candidates": missing_ar,
            "batch_sizes": _parse_csv_ints(args.batch_sizes),
            "ar_k_values": _parse_csv_ints(args.ar_k_values),
            "ddtree_tree_budgets": _parse_csv_ints(args.ddtree_tree_budgets),
            "ddtree_frontier_counts": _parse_csv_ints(args.ddtree_frontier_counts),
            "main_output_len": args.main_output_len,
            "extra_output_len": args.extra_output_len,
            "max_prompts": args.max_prompts,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "main_decomposition_rows": main_table_rows,
        "appendix_equivalence": appendix_equivalence,
        "extra_regime_rows": extra_rows,
        "recommendation": recommendation,
        "artifacts": {
            "final_tables_md": str((artifact_root / "final_tables.md").resolve()),
            "appendix_dflash_md": str((artifact_root / "appendix_dflash_diagnostics.md").resolve()),
            "appendix_extra_md": str((artifact_root / "appendix_extra_regime.md").resolve()),
            "oracle_ceiling_png": str((artifact_root / "figure_oracle_ceiling.png").resolve()),
            "normalized_speedup_png": str((artifact_root / "figure_normalized_speedup.png").resolve()),
            "budget_frontier_png": str((artifact_root / "figure_budget_frontier.png").resolve()),
            "error_bars_png": str((artifact_root / "figure_error_bars.png").resolve()),
        },
    }
    _write_json(artifact_root / "final_summary.json", summary)
    return summary


def main():
    args = parse_args()
    artifact_root = Path(args.output_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    ar_candidates, missing_ar = _parse_ar_draft_candidates(args.ar_draft_candidates)
    if not ar_candidates:
        raise RuntimeError("No AR draft candidates are available. Provide --ar-draft-candidates or set SSD_HF_CACHE.")
    if not args.target or not args.training_metadata or not args.dflash_draft or not args.dflash_predictor:
        raise RuntimeError("--target, --training-metadata, --dflash-draft, and --dflash-predictor are required")

    port_offset = 0
    ar_main_results = []
    ar_extra_results = []
    for label, draft_path in ar_candidates.items():
        ar_main_summary = _prepare_ar_run(
            args,
            artifact_root,
            label,
            draft_path,
            args.main_output_len,
            AR_MODES,
            port_offset,
        )
        port_offset += 100
        ar_extra_summary = _prepare_ar_run(
            args,
            artifact_root,
            label,
            draft_path,
            args.extra_output_len,
            ["ar_async_exact_off_normal", "ar_async_exact_on_oracle"],
            port_offset,
        )
        port_offset += 100
        for row in ar_main_summary["results"]:
            row = dict(row)
            row["draft_label"] = label
            row["output_len"] = args.main_output_len
            ar_main_results.append(row)
        for row in ar_extra_summary["results"]:
            row = dict(row)
            row["draft_label"] = label
            row["output_len"] = args.extra_output_len
            ar_extra_results.append(row)

    dflash_main_summary = _prepare_dflash_run(args, artifact_root, args.main_output_len, long_regime=False, port_offset=port_offset)
    port_offset += 100
    dflash_extra_summary = _prepare_dflash_run(args, artifact_root, args.extra_output_len, long_regime=True, port_offset=port_offset)
    port_offset += 100
    ddtree_main_summary = _prepare_ddtree_run(args, artifact_root, args.main_output_len, long_regime=False, port_offset=port_offset)
    port_offset += 100
    ddtree_extra_summary = _prepare_ddtree_run(args, artifact_root, args.extra_output_len, long_regime=True, port_offset=port_offset)

    dflash_main_results = dflash_main_summary["matrix_a"]["results"]
    dflash_main_matrix_b = dflash_main_summary["matrix_b"]["results"] if dflash_main_summary.get("matrix_b") else []
    dflash_extra_results = dflash_extra_summary["matrix_a"]["results"]
    ddtree_main_results = ddtree_main_summary["results"]
    ddtree_extra_results = ddtree_extra_summary["results"]

    summary = _generate_artifacts(
        args=args,
        artifact_root=artifact_root,
        ar_candidates=ar_candidates,
        missing_ar=missing_ar,
        ar_main_results=ar_main_results,
        ar_extra_results=ar_extra_results,
        dflash_main_results=dflash_main_results,
        dflash_main_matrix_b=dflash_main_matrix_b,
        dflash_extra_results=dflash_extra_results,
        ddtree_main_results=ddtree_main_results,
        ddtree_extra_results=ddtree_extra_results,
    )
    print(json.dumps(summary["recommendation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
