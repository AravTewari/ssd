import argparse
import json
import math
import random
from html import escape
from pathlib import Path


FAMILY_COLORS = {
    "AR": "#2563eb",
    "DFLASH": "#dc2626",
    "DDTREE": "#059669",
}

MODE_COLORS = {
    "exact_dflash": "#334155",
    "dflash_ssd_predicted_off_oracle": "#2563eb",
    "dflash_ssd_predicted_on_oracle": "#059669",
    "dflash_ssd_predicted_on_normal": "#dc2626",
}

MODE_LABELS = {
    "exact_dflash": "Exact DFlash",
    "dflash_ssd_predicted_off_oracle": "Pred+Off+Oracle",
    "dflash_ssd_predicted_on_oracle": "Pred+On+Oracle",
    "dflash_ssd_predicted_on_normal": "Pred+On+Normal",
}

MARKERS = {
    "AR": "circle",
    "DFLASH": "diamond",
    "DDTREE": "square",
}

MPL_MARKERS = {
    "AR": "o",
    "DFLASH": "D",
    "DDTREE": "s",
}

ALLOWED_BATCHES = {1, 2, 4, 8, 16, 32}


def parse_args():
    parser = argparse.ArgumentParser(description="Render final wrap-up figures from saved evaluation artifacts")
    parser.add_argument("--artifact-root", type=str, required=True, help="Final wrap-up artifact directory")
    parser.add_argument("--ar-label", type=str, default=None, help="AR candidate label under runs/ar (defaults to auto-detect)")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _detect_ar_label(artifact_root: Path) -> str:
    runs_dir = artifact_root / "runs" / "ar"
    labels = sorted(child.name for child in runs_dir.iterdir() if child.is_dir())
    if not labels:
        raise RuntimeError(f"No AR runs found under {runs_dir}")
    return labels[0]


def _attach_ar_artifact_dirs(results: list[dict], artifact_root: Path, label: str, output_len: int):
    base = artifact_root / "runs" / "ar" / label / f"len{output_len}" / "summary_artifacts"
    for row in results:
        row["_artifact_dir"] = base / row["mode"] / f"k{int(row['k'])}_b{int(row['batch_size'])}"


def _attach_dflash_artifact_dirs(matrix_a: list[dict], matrix_b: list[dict], artifact_root: Path, regime: str):
    base = artifact_root / "runs" / "dflash" / regime / "summary_artifacts"
    for row in matrix_a:
        row["_artifact_dir"] = base / "matrix_a" / f"{row['mode']}_b{int(row['batch_size'])}"
    for row in matrix_b:
        row["_artifact_dir"] = base / "matrix_b" / f"{row['fanout_template_name']}_b{int(row['batch_size'])}"


def _attach_ddtree_artifact_dirs(results: list[dict], artifact_root: Path, regime: str):
    base = artifact_root / "runs" / "ddtree" / regime / "summary_artifacts"
    for row in results:
        row["_artifact_dir"] = (
            base
            / row["mode"]
            / f"b{int(row['batch_size'])}_tb{int(row['tree_budget'])}_fc{int(row['frontier_count'])}"
        )


def _bootstrap_ci(values: list[float], samples: int, seed: int) -> dict:
    if not values:
        return {"mean": None, "low": None, "high": None}
    rng = random.Random(seed)
    draws = []
    count = len(values)
    for _ in range(samples):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        draws.append(sum(sample) / count)
    draws.sort()
    low_idx = int(0.025 * (len(draws) - 1))
    high_idx = int(0.975 * (len(draws) - 1))
    return {
        "mean": sum(values) / count,
        "low": draws[low_idx],
        "high": draws[high_idx],
    }


def _annotate_ci(rows: list[dict], samples: int, seed: int):
    for idx, row in enumerate(rows):
        metrics_path = Path(row["_artifact_dir"]) / "prompt_group_metrics.json"
        if not metrics_path.exists():
            row["_ci"] = {
                "throughput_tok_s": {"mean": None, "low": None, "high": None},
                "accepted_suffix_mean": {"mean": None, "low": None, "high": None},
            }
            continue
        group_metrics = _load_json(metrics_path)
        row["_ci"] = {
            "throughput_tok_s": _bootstrap_ci(
                [item["throughput_tok_s"] for item in group_metrics.values() if item.get("throughput_tok_s") is not None],
                samples,
                seed + idx * 2,
            ),
            "accepted_suffix_mean": _bootstrap_ci(
                [item["accepted_suffix_mean"] for item in group_metrics.values() if item.get("accepted_suffix_mean") is not None],
                samples,
                seed + idx * 2 + 1,
            ),
        }


def _row_value(row: dict, metric: str) -> float | None:
    ci = row.get("_ci", {}).get(metric, {})
    if ci.get("mean") is not None:
        return ci["mean"]
    return row.get(metric)


def _row_ci(row: dict, metric: str) -> tuple[float | None, float | None]:
    ci = row.get("_ci", {}).get(metric, {})
    if ci.get("low") is None or ci.get("high") is None:
        return None, None
    return ci["low"], ci["high"]


def _group_by_batch(rows: list[dict]) -> dict[int, list[dict]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["batch_size"]), []).append(row)
    return grouped


def _filter_allowed_batches(rows: list[dict]) -> list[dict]:
    return [row for row in rows if int(row["batch_size"]) in ALLOWED_BATCHES]


def _sorted_batches_from_rows(*row_groups: list[dict]) -> list[int]:
    batches = {
        int(row["batch_size"])
        for rows in row_groups
        for row in rows
        if row.get("batch_size") is not None
    }
    return sorted(batches)


def _best_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: row["throughput_tok_s"])


def _extract_best_ar(rows: list[dict], mode: str) -> list[dict]:
    grouped = _group_by_batch([row for row in rows if row["mode"] == mode])
    return [_best_row(grouped[b]) for b in sorted(grouped)]


def _extract_best_ddtree(rows: list[dict], mode: str) -> list[dict]:
    grouped = _group_by_batch([row for row in rows if row["mode"] == mode])
    return [_best_row(grouped[b]) for b in sorted(grouped)]


def _index_by_batch(rows: list[dict]) -> dict[int, dict]:
    return {int(row["batch_size"]): row for row in rows}


def _normalized_speedups(oracle_rows: list[dict], exact_off_rows: list[dict]) -> list[dict]:
    exact_by_batch = _index_by_batch(exact_off_rows)
    out = []
    for row in oracle_rows:
        base = exact_by_batch[int(row["batch_size"])]
        point = _row_value(row, "throughput_tok_s")
        base_point = _row_value(base, "throughput_tok_s")
        out.append(
            {
                "family": row.get("family"),
                "batch_size": int(row["batch_size"]),
                "ratio": point / max(base_point, 1e-6),
            }
        )
    return out


def _is_dominated(point: dict, others: list[dict]) -> bool:
    for other in others:
        if other is point:
            continue
        if (
            other["x"] >= point["x"]
            and other["y"] >= point["y"]
            and (other["x"] > point["x"] or other["y"] > point["y"])
        ):
            return True
    return False


def _pareto_frontier(points: list[dict]) -> list[dict]:
    frontier = [point for point in points if not _is_dominated(point, points)]
    return sorted(frontier, key=lambda point: (point["x"], point["y"]))


def _nice_ticks(vmin: float, vmax: float, count: int = 5) -> list[float]:
    if math.isclose(vmin, vmax):
        return [vmin]
    raw_step = max((vmax - vmin) / max(count - 1, 1), 1e-9)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    step = nice * magnitude
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks = []
    value = start
    for _ in range(100):
        ticks.append(round(value, 10))
        value += step
        if value > end + 1e-9:
            break
    return ticks


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d1d5db",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
            "grid.linestyle": "-",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt, Line2D, Patch


def _save_figure(fig, base_path: Path):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(base_path.with_suffix(f".{ext}"))
    fig.clf()


def _ci_errorbars(rows: list[dict], metric: str):
    means = []
    lower = []
    upper = []
    for row in rows:
        mean = _row_value(row, metric)
        lo, hi = _row_ci(row, metric)
        means.append(mean)
        if lo is None or hi is None or mean is None:
            lower.append(0.0)
            upper.append(0.0)
        else:
            lower.append(max(mean - lo, 0.0))
            upper.append(max(hi - mean, 0.0))
    return means, [lower, upper]


def _render_oracle_ceiling_mpl(
    base_path: Path,
    main_ar: list[dict],
    main_dflash: list[dict],
    main_ddtree: list[dict],
    extra_ar: list[dict],
    extra_dflash: list[dict],
    extra_ddtree: list[dict],
):
    plt, Line2D, _ = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True, constrained_layout=True)
    panel_specs = [
        ("Output length = 32", axes[0], main_ar, main_dflash, main_ddtree),
        ("Output length = 128", axes[1], extra_ar, extra_dflash, extra_ddtree),
    ]
    families = [("AR", main_ar), ("DFLASH", main_dflash), ("DDTREE", main_ddtree)]
    handles = [
        Line2D([0], [0], color=FAMILY_COLORS[family], marker=MPL_MARKERS[family], lw=2.2, ms=6, label=family)
        for family, _ in families
    ]
    for title, ax, ar_rows, dflash_rows, ddtree_rows in panel_specs:
        for family, rows in [("AR", ar_rows), ("DFLASH", dflash_rows), ("DDTREE", ddtree_rows)]:
            rows = sorted(rows, key=lambda row: row["batch_size"])
            xs = [int(row["batch_size"]) for row in rows]
            ys, yerr = _ci_errorbars(rows, "throughput_tok_s")
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                color=FAMILY_COLORS[family],
                marker=MPL_MARKERS[family],
                linewidth=2.2,
                markersize=6,
                capsize=3,
                elinewidth=1.4,
                markeredgecolor="white",
                markeredgewidth=0.7,
            )
        ax.set_title(title)
        ax.set_xlabel("Batch size $b$")
        panel_batches = _sorted_batches_from_rows(ar_rows, dflash_rows, ddtree_rows)
        ax.set_xscale("log", base=2)
        ax.set_xticks(panel_batches)
        ax.set_xticklabels([str(batch) for batch in panel_batches])
        ax.set_xlim(min(panel_batches) / 1.15, max(panel_batches) * 1.15)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    axes[0].set_ylabel("Throughput (tok/s)")
    axes[0].legend(handles=handles, loc="upper left", frameon=False, ncol=1)
    fig.suptitle("Oracle Ceiling Comparison", y=1.02, fontsize=14, fontweight="bold")
    _save_figure(fig, base_path)
    plt.close(fig)


def _render_normalized_headroom_mpl(base_path: Path, main_bars: list[dict], extra_bars: list[dict]):
    plt, _, Patch = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), sharey=True, constrained_layout=True)
    panels = [
        ("Output length = 32", axes[0], main_bars),
        ("Output length = 128", axes[1], extra_bars),
    ]
    family_order = ["AR", "DFLASH", "DDTREE"]
    width = 0.22
    legend_handles = [Patch(facecolor=FAMILY_COLORS[f], edgecolor="none", label=f) for f in family_order]
    for title, ax, rows in panels:
        batch_order = _sorted_batches_from_rows(rows)
        grouped = _group_by_batch(rows)
        x_positions = list(range(len(batch_order)))
        for j, family in enumerate(family_order):
            values = []
            for batch in batch_order:
                row = next(item for item in grouped[batch] if item["family"] == family)
                values.append(row["ratio"])
            offsets = [x + (j - 1) * width for x in x_positions]
            ax.bar(offsets, values, width=width, color=FAMILY_COLORS[family], alpha=0.9, label=family)
        ax.axhline(1.0, color="#6b7280", linestyle="--", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Batch size $b$")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(b) for b in batch_order])
        ax.set_xlim(-0.6, len(batch_order) - 0.4)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
    axes[0].set_ylabel(r"Normalized headroom: $\mathrm{tok/s}_{oracle} / \mathrm{tok/s}_{exact\ off}$")
    axes[0].legend(handles=legend_handles, loc="upper left", frameon=False)
    fig.suptitle("Normalized SSD Headroom", y=1.02, fontsize=14, fontweight="bold")
    _save_figure(fig, base_path)
    plt.close(fig)


def _render_frontier_mpl(base_path: Path, ar_rows: list[dict], dflash_rows: list[dict], ddtree_rows: list[dict]):
    plt, Line2D, _ = _mpl()
    batch_order = _sorted_batches_from_rows(ar_rows, dflash_rows, ddtree_rows)
    ncols = 4
    nrows = math.ceil(len(batch_order) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16.0, 3.8 * nrows),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, batch in zip(axes, batch_order):
        points = []
        for family, rows in [
            ("AR", [row for row in ar_rows if int(row["batch_size"]) == batch]),
            ("DFLASH", [row for row in dflash_rows if int(row["batch_size"]) == batch]),
            ("DDTREE", [row for row in ddtree_rows if int(row["batch_size"]) == batch]),
        ]:
            for row in rows:
                points.append(
                    {
                        "family": family,
                        "x": _row_value(row, "accepted_suffix_mean"),
                        "y": _row_value(row, "throughput_tok_s"),
                    }
                )
        for family in ["AR", "DFLASH", "DDTREE"]:
            family_points = [point for point in points if point["family"] == family]
            ax.scatter(
                [point["x"] for point in family_points],
                [point["y"] for point in family_points],
                s=55 if family != "DFLASH" else 80,
                marker=MPL_MARKERS[family],
                c=FAMILY_COLORS[family],
                edgecolors="white",
                linewidths=0.7,
                alpha=0.95,
                zorder=3,
            )
        frontier = _pareto_frontier(points)
        if frontier:
            ax.plot(
                [point["x"] for point in frontier],
                [point["y"] for point in frontier],
                linestyle="--",
                linewidth=1.8,
                color="#111827",
                zorder=2,
            )
            ax.scatter(
                [point["x"] for point in frontier],
                [point["y"] for point in frontier],
                s=110,
                facecolors="none",
                edgecolors="#111827",
                linewidths=1.4,
                zorder=4,
            )
        ax.set_title(f"$b={batch}$")
        ax.set_xlabel("Accepted suffix")
        ax.grid(True, axis="both")
    for idx, ax in enumerate(axes):
        if idx >= len(batch_order):
            ax.axis("off")
        elif idx % ncols == 0:
            ax.set_ylabel("Throughput (tok/s)")
    legend_handles = [
        Line2D([0], [0], color="none", marker=MPL_MARKERS[family], markerfacecolor=FAMILY_COLORS[family], markeredgecolor="white", markeredgewidth=0.7, markersize=7, label=family)
        for family in ["AR", "DFLASH", "DDTREE"]
    ]
    legend_handles.append(Line2D([0], [0], color="#111827", linestyle="--", linewidth=1.8, label="Pareto frontier"))
    axes[0].legend(handles=legend_handles, loc="best", frameon=False)
    fig.suptitle("Throughput vs Accepted-Suffix Frontier (Exact-On-Oracle, output length = 32)", y=1.04, fontsize=14, fontweight="bold")
    _save_figure(fig, base_path)
    plt.close(fig)


def _render_dflash_failure_mpl(base_path: Path, matrix_a_rows: list[dict]):
    plt, _, Patch = _mpl()
    fig, ax = plt.subplots(figsize=(13.6, 4.8), constrained_layout=True)
    ax2 = ax.twinx()
    modes = [
        "exact_dflash",
        "dflash_ssd_predicted_off_oracle",
        "dflash_ssd_predicted_on_oracle",
        "dflash_ssd_predicted_on_normal",
    ]
    rows = [row for row in matrix_a_rows if row["mode"] in modes]
    batches = _sorted_batches_from_rows(rows)
    lookup = {(row["mode"], int(row["batch_size"])): row for row in rows}
    x_positions = list(range(len(batches)))
    width = 0.16
    bar_handles = []
    for j, mode in enumerate(modes):
        values = [lookup[(mode, batch)]["throughput_tok_s"] for batch in batches]
        offsets = [x + (j - 1.5) * width for x in x_positions]
        bars = ax.bar(offsets, values, width=width, color=MODE_COLORS[mode], alpha=0.9, label=MODE_LABELS[mode])
        bar_handles.append(Patch(facecolor=MODE_COLORS[mode], label=MODE_LABELS[mode]))
    cache_modes = [
        ("dflash_ssd_predicted_on_oracle", "#065f46", "-", "Cache hit: Pred+On+Oracle"),
        ("dflash_ssd_predicted_on_normal", "#7f1d1d", "--", "Cache hit: Pred+On+Normal"),
    ]
    line_handles = []
    for mode, color, style, label in cache_modes:
        values = [lookup[(mode, batch)]["cache_hit_mean"] for batch in batches]
        xs = [x + (modes.index(mode) - 1.5) * width for x in x_positions]
        handle = ax2.plot(xs, values, color=color, linestyle=style, linewidth=2.0, marker="o", markersize=5, label=label)[0]
        line_handles.append(handle)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size $b$")
    ax.set_ylabel("Throughput (tok/s)")
    ax2.set_ylabel("Cache hit rate")
    ax2.set_ylim(0.0, 1.02)
    ax.set_xlim(-0.7, len(batches) - 0.3)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.set_title("Throughput bars with cache-hit overlay", fontsize=12)
    ax.legend(
        handles=bar_handles + line_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        frameon=False,
        ncol=2,
        columnspacing=1.2,
        handlelength=2.4,
    )
    fig.suptitle("Appendix: DFlash Branch-Cache Failure", y=1.03, fontsize=14, fontweight="bold")
    _save_figure(fig, base_path)
    plt.close(fig)


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.items = []

    def add(self, item: str):
        self.items.append(item)

    def line(self, x1, y1, x2, y2, stroke="#000", width=1.5, dash=None, opacity=1.0):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width:.2f}" opacity="{opacity:.3f}"{dash_attr} />'
        )

    def rect(self, x, y, w, h, fill="none", stroke="none", width=1.0, opacity=1.0, rx=0):
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width:.2f}" opacity="{opacity:.3f}" rx="{rx}" />'
        )

    def circle(self, cx, cy, r, fill="none", stroke="none", width=1.0, opacity=1.0):
        self.add(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width:.2f}" opacity="{opacity:.3f}" />'
        )

    def polygon(self, points, fill="none", stroke="none", width=1.0, opacity=1.0):
        payload = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(
            f'<polygon points="{payload}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width:.2f}" opacity="{opacity:.3f}" />'
        )

    def polyline(self, points, stroke="#000", width=1.5, fill="none", dash=None, opacity=1.0):
        payload = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<polyline points="{payload}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width:.2f}" opacity="{opacity:.3f}"{dash_attr} />'
        )

    def text(self, x, y, text, size=14, fill="#111", anchor="start", weight="normal", rotate=None):
        rotate_attr = f' transform="rotate({rotate:.2f} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{fill}" '
            f'font-family="Helvetica, Arial, sans-serif" text-anchor="{anchor}" '
            f'font-weight="{weight}"{rotate_attr}>{escape(str(text))}</text>'
        )

    def save(self, path: Path):
        content = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
            '<rect width="100%" height="100%" fill="white" />',
            *self.items,
            "</svg>",
        ]
        _write_text(path, "\n".join(content) + "\n")


def _draw_marker(svg: Svg, family: str, x: float, y: float, size: float, fill: str, stroke: str = "#ffffff", stroke_width: float = 1.5):
    shape = MARKERS[family]
    if shape == "circle":
        svg.circle(x, y, size, fill=fill, stroke=stroke, width=stroke_width)
    elif shape == "square":
        svg.rect(x - size, y - size, 2 * size, 2 * size, fill=fill, stroke=stroke, width=stroke_width)
    elif shape == "diamond":
        svg.polygon(
            [(x, y - size), (x + size, y), (x, y + size), (x - size, y)],
            fill=fill,
            stroke=stroke,
            width=stroke_width,
        )


def _draw_axes(svg: Svg, x: float, y: float, w: float, h: float, y_ticks: list[float], y_min: float, y_max: float, title: str, show_y_labels: bool = True):
    svg.rect(x, y, w, h, fill="none", stroke="#111827", width=1.2)
    for tick in y_ticks:
        yy = y + h - (tick - y_min) / max(y_max - y_min, 1e-9) * h
        svg.line(x, yy, x + w, yy, stroke="#e5e7eb", width=1)
        if show_y_labels:
            label = f"{tick:.0f}" if abs(tick) >= 10 or math.isclose(tick, round(tick)) else f"{tick:.2f}"
            svg.text(x - 10, yy + 4, label, size=12, anchor="end", fill="#374151")
    svg.text(x + w / 2, y - 12, title, size=16, anchor="middle", weight="bold")


def _render_oracle_ceiling_figure(
    out_path: Path,
    main_ar: list[dict],
    main_dflash: list[dict],
    main_ddtree: list[dict],
    extra_ar: list[dict],
    extra_dflash: list[dict],
    extra_ddtree: list[dict],
):
    svg = Svg(1240, 520)
    panels = [
        ("Output length 32", 90, 80, 470, 350, main_ar, main_dflash, main_ddtree),
        ("Output length 128", 690, 80, 470, 350, extra_ar, extra_dflash, extra_ddtree),
    ]
    family_order = [("AR", main_ar), ("DFLASH", main_dflash), ("DDTREE", main_ddtree)]
    all_rows = [*main_ar, *main_dflash, *main_ddtree, *extra_ar, *extra_dflash, *extra_ddtree]
    y_highs = []
    for row in all_rows:
        low, high = _row_ci(row, "throughput_tok_s")
        point = _row_value(row, "throughput_tok_s")
        y_highs.append(high if high is not None else point)
    global_y_max = max(y_highs) * 1.08
    x_batches = [1, 2, 4]
    for title, x, y, w, h, ar_rows, dflash_rows, ddtree_rows in panels:
        y_ticks = _nice_ticks(0.0, global_y_max, count=6)
        _draw_axes(svg, x, y, w, h, y_ticks, 0.0, max(y_ticks), title, show_y_labels=True)
        for batch in x_batches:
            xx = x + (x_batches.index(batch) + 0.5) / len(x_batches) * w
            svg.line(xx, y + h, xx, y + h + 6, stroke="#111827", width=1.2)
            svg.text(xx, y + h + 24, str(batch), size=13, anchor="middle", fill="#111827")
        for family, rows in [("AR", ar_rows), ("DFLASH", dflash_rows), ("DDTREE", ddtree_rows)]:
            color = FAMILY_COLORS[family]
            points = []
            for row in sorted(rows, key=lambda item: item["batch_size"]):
                xx = x + (x_batches.index(int(row["batch_size"])) + 0.5) / len(x_batches) * w
                point = _row_value(row, "throughput_tok_s")
                yy = y + h - point / max(y_ticks) * h
                low, high = _row_ci(row, "throughput_tok_s")
                if low is not None and high is not None:
                    y_low = y + h - low / max(y_ticks) * h
                    y_high = y + h - high / max(y_ticks) * h
                    svg.line(xx, y_low, xx, y_high, stroke=color, width=1.8)
                    svg.line(xx - 5, y_low, xx + 5, y_low, stroke=color, width=1.8)
                    svg.line(xx - 5, y_high, xx + 5, y_high, stroke=color, width=1.8)
                points.append((xx, yy))
            svg.polyline(points, stroke=color, width=2.8)
            for xx, yy in points:
                _draw_marker(svg, family, xx, yy, 6, fill=color)
    svg.text(620, 34, "Oracle Ceiling Comparison", size=22, anchor="middle", weight="bold")
    svg.text(620, 485, "Batch size b", size=16, anchor="middle")
    svg.text(28, 255, "Throughput (tok/s)", size=16, anchor="middle", rotate=-90)
    legend_x = 945
    legend_y = 36
    for idx, family in enumerate(["AR", "DFLASH", "DDTREE"]):
        yy = legend_y + idx * 22
        svg.line(legend_x, yy, legend_x + 28, yy, stroke=FAMILY_COLORS[family], width=2.8)
        _draw_marker(svg, family, legend_x + 14, yy, 5.5, fill=FAMILY_COLORS[family])
        svg.text(legend_x + 40, yy + 5, family, size=13)
    svg.save(out_path)


def _render_normalized_headroom_figure(
    out_path: Path,
    main_bars: list[dict],
    extra_bars: list[dict],
):
    svg = Svg(1240, 520)
    panels = [
        ("Output length 32", 90, 80, 470, 350, main_bars),
        ("Output length 128", 690, 80, 470, 350, extra_bars),
    ]
    max_ratio = max(item["ratio"] for item in [*main_bars, *extra_bars]) * 1.1
    y_max = max(max_ratio, 1.25)
    y_ticks = _nice_ticks(0.0, y_max, count=6)
    bar_order = ["AR", "DFLASH", "DDTREE"]
    batches = [1, 2, 4]
    for title, x, y, w, h, bars in panels:
        _draw_axes(svg, x, y, w, h, y_ticks, 0.0, max(y_ticks), title, show_y_labels=True)
        baseline_y = y + h - 1.0 / max(y_ticks) * h
        svg.line(x, baseline_y, x + w, baseline_y, stroke="#9ca3af", width=1.5, dash="6 6")
        cluster_w = w / len(batches)
        bar_w = cluster_w * 0.18
        by_batch = _group_by_batch(bars)
        for i, batch in enumerate(batches):
            center = x + (i + 0.5) * cluster_w
            svg.text(center, y + h + 24, str(batch), size=13, anchor="middle")
            for j, family in enumerate(bar_order):
                row = next(item for item in by_batch[batch] if item["family"] == family)
                xx = center + (j - 1) * (bar_w + 10) - bar_w / 2
                yy = y + h - row["ratio"] / max(y_ticks) * h
                svg.rect(xx, yy, bar_w, y + h - yy, fill=FAMILY_COLORS[family], stroke="none", opacity=0.9)
        legend_x = x + w - 110
        legend_y = y + 18
        for idx, family in enumerate(bar_order):
            yy = legend_y + idx * 22
            svg.rect(legend_x, yy - 10, 14, 14, fill=FAMILY_COLORS[family], stroke="none")
            svg.text(legend_x + 22, yy + 1, family, size=13)
    svg.text(620, 34, "Normalized SSD Headroom", size=22, anchor="middle", weight="bold")
    svg.text(620, 485, "Batch size b", size=16, anchor="middle")
    svg.text(28, 255, "Oracle tok/s divided by Exact-Off tok/s", size=16, anchor="middle", rotate=-90)
    svg.save(out_path)


def _render_frontier_figure(out_path: Path, ar_rows: list[dict], dflash_rows: list[dict], ddtree_rows: list[dict]):
    svg = Svg(1380, 470)
    panels = []
    for idx, batch in enumerate([1, 2, 4]):
        panels.append(
            (
                f"b = {batch}",
                80 + idx * 430,
                70,
                360,
                320,
                [row for row in ar_rows if int(row["batch_size"]) == batch],
                [row for row in dflash_rows if int(row["batch_size"]) == batch],
                [row for row in ddtree_rows if int(row["batch_size"]) == batch],
            )
        )
    for title, x, y, w, h, ar_panel, dflash_panel, ddtree_panel in panels:
        points = []
        for family, rows in [("AR", ar_panel), ("DFLASH", dflash_panel), ("DDTREE", ddtree_panel)]:
            for row in rows:
                points.append(
                    {
                        "family": family,
                        "row": row,
                        "x": _row_value(row, "accepted_suffix_mean"),
                        "y": _row_value(row, "throughput_tok_s"),
                    }
                )
        x_vals = [point["x"] for point in points if point["x"] is not None]
        y_vals = [point["y"] for point in points if point["y"] is not None]
        x_min = min(x_vals) * 0.95
        x_max = max(x_vals) * 1.05
        y_min = 0.0
        y_max = max(y_vals) * 1.08
        x_ticks = _nice_ticks(x_min, x_max, count=5)
        y_ticks = _nice_ticks(y_min, y_max, count=5)
        _draw_axes(svg, x, y, w, h, y_ticks, y_min, max(y_ticks), title, show_y_labels=True)
        for tick in x_ticks:
            xx = x + (tick - x_min) / max(x_max - x_min, 1e-9) * w
            svg.line(xx, y + h, xx, y + h + 6, stroke="#111827", width=1.2)
            label = f"{tick:.2f}" if tick < 10 else f"{tick:.1f}"
            svg.text(xx, y + h + 24, label, size=11, anchor="middle", fill="#374151")
            svg.line(xx, y, xx, y + h, stroke="#f3f4f6", width=1)
        frontier = _pareto_frontier(points)
        frontier_points = []
        for point in points:
            xx = x + (point["x"] - x_min) / max(x_max - x_min, 1e-9) * w
            yy = y + h - point["y"] / max(y_ticks) * h
            point["_svg"] = (xx, yy)
            _draw_marker(svg, point["family"], xx, yy, 5.5, fill=FAMILY_COLORS[point["family"]], stroke="#ffffff", stroke_width=1.3)
        for point in frontier:
            xx, yy = point["_svg"]
            _draw_marker(svg, point["family"], xx, yy, 8, fill="none", stroke="#111827", stroke_width=1.8)
            frontier_points.append((xx, yy))
        if len(frontier_points) >= 2:
            svg.polyline(frontier_points, stroke="#111827", width=2.0, dash="6 4", fill="none")
    svg.text(690, 30, "Throughput vs Accepted-Suffix Frontier (Exact-On-Oracle, Output length 32)", size=22, anchor="middle", weight="bold")
    svg.text(690, 438, "Accepted suffix", size=16, anchor="middle")
    svg.text(28, 230, "Throughput (tok/s)", size=16, anchor="middle", rotate=-90)
    legend_x = 1080
    legend_y = 30
    for idx, family in enumerate(["AR", "DFLASH", "DDTREE"]):
        yy = legend_y + idx * 22
        _draw_marker(svg, family, legend_x, yy, 6, fill=FAMILY_COLORS[family])
        svg.text(legend_x + 18, yy + 5, family, size=13)
    svg.line(1080, 100, 1110, 100, stroke="#111827", width=2.0, dash="6 4")
    svg.text(1118, 105, "Pareto frontier", size=13)
    svg.save(out_path)


def _render_dflash_failure_figure(out_path: Path, matrix_a_rows: list[dict]):
    svg = Svg(1240, 540)
    x = 90
    y = 80
    w = 920
    h = 340
    batches = [1, 2, 4]
    modes = [
        "exact_dflash",
        "dflash_ssd_predicted_off_oracle",
        "dflash_ssd_predicted_on_oracle",
        "dflash_ssd_predicted_on_normal",
    ]
    rows = [row for row in matrix_a_rows if row["mode"] in modes]
    y_max = max(row["throughput_tok_s"] for row in rows) * 1.12
    y_ticks = _nice_ticks(0.0, y_max, count=6)
    _draw_axes(svg, x, y, w, h, y_ticks, 0.0, max(y_ticks), "DFlash Branch-Cache Failure", show_y_labels=True)
    right_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for tick in right_ticks:
        yy = y + h - tick * h
        svg.text(x + w + 10, yy + 4, f"{tick:.2f}", size=12, fill="#6b7280")
    svg.text(x + w + 52, y + h / 2, "Cache hit rate", size=15, anchor="middle", rotate=90, fill="#6b7280")
    cluster_w = w / len(batches)
    bar_w = cluster_w * 0.15
    centers = {}
    by_mode_batch = {(row["mode"], int(row["batch_size"])): row for row in rows}
    for i, batch in enumerate(batches):
        center = x + (i + 0.5) * cluster_w
        svg.text(center, y + h + 24, str(batch), size=13, anchor="middle")
        for j, mode in enumerate(modes):
            xx = center + (j - 1.5) * (bar_w + 10) - bar_w / 2
            row = by_mode_batch[(mode, batch)]
            yy = y + h - row["throughput_tok_s"] / max(y_ticks) * h
            svg.rect(xx, yy, bar_w, y + h - yy, fill=MODE_COLORS[mode], stroke="none", opacity=0.9)
            centers[(mode, batch)] = xx + bar_w / 2
    for mode, color, dash in [
        ("dflash_ssd_predicted_on_oracle", "#065f46", "none"),
        ("dflash_ssd_predicted_on_normal", "#7f1d1d", "6 4"),
    ]:
        line_points = []
        for batch in batches:
            row = by_mode_batch[(mode, batch)]
            hit = row.get("cache_hit_mean")
            xx = centers[(mode, batch)]
            yy = y + h - hit * h
            line_points.append((xx, yy))
        svg.polyline(line_points, stroke=color, width=2.2, dash=None if dash == "none" else dash)
        for xx, yy in line_points:
            svg.circle(xx, yy, 4.5, fill=color, stroke="white", width=1.2)
    svg.text(620, 34, "Appendix: DFlash Branch-Cache Failure", size=22, anchor="middle", weight="bold")
    svg.text(550, 490, "Batch size b", size=16, anchor="middle")
    svg.text(28, 250, "Throughput (tok/s)", size=16, anchor="middle", rotate=-90)
    legend_x = 1020
    legend_y = 80
    for idx, mode in enumerate(modes):
        yy = legend_y + idx * 22
        svg.rect(legend_x, yy - 10, 14, 14, fill=MODE_COLORS[mode], stroke="none")
        svg.text(legend_x + 22, yy + 1, MODE_LABELS[mode], size=12)
    for idx, (label, color, dash) in enumerate([
        ("Cache hit: Pred+On+Oracle", "#065f46", None),
        ("Cache hit: Pred+On+Normal", "#7f1d1d", "6 4"),
    ]):
        yy = legend_y + 100 + idx * 22
        svg.line(legend_x, yy, legend_x + 24, yy, stroke=color, width=2.2, dash=dash)
        svg.circle(legend_x + 12, yy, 4.5, fill=color, stroke="white", width=1.2)
        svg.text(legend_x + 34, yy + 4, label, size=12)
    svg.save(out_path)


def main():
    args = parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    ar_label = args.ar_label or _detect_ar_label(artifact_root)

    ar_main_summary = _load_json(artifact_root / "runs" / "ar" / ar_label / "len32" / "summary.json")
    ar_extra_summary = _load_json(artifact_root / "runs" / "ar" / ar_label / "len128" / "summary.json")
    dflash_main_summary = _load_json(artifact_root / "runs" / "dflash" / "main" / "summary.json")
    dflash_extra_summary = _load_json(artifact_root / "runs" / "dflash" / "extra" / "summary.json")
    ddtree_main_summary = _load_json(artifact_root / "runs" / "ddtree" / "main" / "summary.json")
    ddtree_extra_summary = _load_json(artifact_root / "runs" / "ddtree" / "extra" / "summary.json")

    ar_main_results = ar_main_summary["results"]
    ar_extra_results = ar_extra_summary["results"]
    dflash_main_matrix_a = dflash_main_summary["matrix_a"]["results"]
    dflash_main_matrix_b = dflash_main_summary["matrix_b"]["results"]
    dflash_extra_matrix_a = dflash_extra_summary["matrix_a"]["results"]
    ddtree_main_results = ddtree_main_summary["results"]
    ddtree_extra_results = ddtree_extra_summary["results"]

    ar_main_results = _filter_allowed_batches(ar_main_results)
    ar_extra_results = _filter_allowed_batches(ar_extra_results)
    dflash_main_matrix_a = _filter_allowed_batches(dflash_main_matrix_a)
    dflash_main_matrix_b = _filter_allowed_batches(dflash_main_matrix_b)
    dflash_extra_matrix_a = _filter_allowed_batches(dflash_extra_matrix_a)
    ddtree_main_results = _filter_allowed_batches(ddtree_main_results)
    ddtree_extra_results = _filter_allowed_batches(ddtree_extra_results)

    _attach_ar_artifact_dirs(ar_main_results, artifact_root, ar_label, 32)
    _attach_ar_artifact_dirs(ar_extra_results, artifact_root, ar_label, 128)
    _attach_dflash_artifact_dirs(dflash_main_matrix_a, dflash_main_matrix_b, artifact_root, "main")
    _attach_dflash_artifact_dirs(dflash_extra_matrix_a, [], artifact_root, "extra")
    _attach_ddtree_artifact_dirs(ddtree_main_results, artifact_root, "main")
    _attach_ddtree_artifact_dirs(ddtree_extra_results, artifact_root, "extra")

    _annotate_ci(ar_main_results, args.bootstrap_samples, args.bootstrap_seed + 0)
    _annotate_ci(ar_extra_results, args.bootstrap_samples, args.bootstrap_seed + 1000)
    _annotate_ci(dflash_main_matrix_a, args.bootstrap_samples, args.bootstrap_seed + 2000)
    _annotate_ci(dflash_main_matrix_b, args.bootstrap_samples, args.bootstrap_seed + 3000)
    _annotate_ci(dflash_extra_matrix_a, args.bootstrap_samples, args.bootstrap_seed + 4000)
    _annotate_ci(ddtree_main_results, args.bootstrap_samples, args.bootstrap_seed + 5000)
    _annotate_ci(ddtree_extra_results, args.bootstrap_samples, args.bootstrap_seed + 6000)

    main_ar_oracle = _extract_best_ar(ar_main_results, "ar_async_exact_on_oracle")
    extra_ar_oracle = _extract_best_ar(ar_extra_results, "ar_async_exact_on_oracle")
    main_ar_exact = _extract_best_ar(ar_main_results, "ar_async_exact_off_normal")
    extra_ar_exact = _extract_best_ar(ar_extra_results, "ar_async_exact_off_normal")

    main_dflash_oracle = sorted(
        [row for row in dflash_main_matrix_a if row["mode"] == "dflash_ssd_exact_on_oracle"],
        key=lambda row: row["batch_size"],
    )
    extra_dflash_oracle = sorted(
        [row for row in dflash_extra_matrix_a if row["mode"] == "dflash_ssd_exact_on_oracle"],
        key=lambda row: row["batch_size"],
    )
    main_dflash_exact = sorted(
        [row for row in dflash_main_matrix_a if row["mode"] == "dflash_ssd_exact_off_normal"],
        key=lambda row: row["batch_size"],
    )
    extra_dflash_exact = sorted(
        [row for row in dflash_extra_matrix_a if row["mode"] == "dflash_ssd_exact_off_normal"],
        key=lambda row: row["batch_size"],
    )

    main_ddtree_oracle = _extract_best_ddtree(ddtree_main_results, "ddtree_ssd_exact_on_oracle")
    extra_ddtree_oracle = _extract_best_ddtree(ddtree_extra_results, "ddtree_ssd_exact_on_oracle")
    main_ddtree_exact = _extract_best_ddtree(ddtree_main_results, "ddtree_ssd_exact_off")
    extra_ddtree_exact = _extract_best_ddtree(ddtree_extra_results, "ddtree_ssd_exact_off")

    for row in main_ar_oracle + extra_ar_oracle:
        row["family"] = "AR"
    for row in main_dflash_oracle + extra_dflash_oracle:
        row["family"] = "DFLASH"
    for row in main_ddtree_oracle + extra_ddtree_oracle:
        row["family"] = "DDTREE"

    main_headroom = (
        _normalized_speedups(main_ar_oracle, main_ar_exact)
        + _normalized_speedups(main_dflash_oracle, main_dflash_exact)
        + _normalized_speedups(main_ddtree_oracle, main_ddtree_exact)
    )
    extra_headroom = (
        _normalized_speedups(extra_ar_oracle, extra_ar_exact)
        + _normalized_speedups(extra_dflash_oracle, extra_dflash_exact)
        + _normalized_speedups(extra_ddtree_oracle, extra_ddtree_exact)
    )

    _render_oracle_ceiling_mpl(
        artifact_root / "figure_oracle_ceiling",
        main_ar_oracle,
        main_dflash_oracle,
        main_ddtree_oracle,
        extra_ar_oracle,
        extra_dflash_oracle,
        extra_ddtree_oracle,
    )
    _render_normalized_headroom_mpl(
        artifact_root / "figure_normalized_speedup",
        main_headroom,
        extra_headroom,
    )
    _render_frontier_mpl(
        artifact_root / "figure_budget_frontier",
        [row for row in ar_main_results if row["mode"] == "ar_async_exact_on_oracle"],
        [row for row in dflash_main_matrix_a if row["mode"] == "dflash_ssd_exact_on_oracle"],
        [row for row in ddtree_main_results if row["mode"] == "ddtree_ssd_exact_on_oracle"],
    )
    _render_dflash_failure_mpl(
        artifact_root / "figure_dflash_branch_cache_failure",
        dflash_main_matrix_a,
    )

    manifest = {
        "oracle_ceiling_png": str((artifact_root / "figure_oracle_ceiling.png").resolve()),
        "oracle_ceiling_pdf": str((artifact_root / "figure_oracle_ceiling.pdf").resolve()),
        "oracle_ceiling_svg": str((artifact_root / "figure_oracle_ceiling.svg").resolve()),
        "normalized_speedup_png": str((artifact_root / "figure_normalized_speedup.png").resolve()),
        "normalized_speedup_pdf": str((artifact_root / "figure_normalized_speedup.pdf").resolve()),
        "normalized_speedup_svg": str((artifact_root / "figure_normalized_speedup.svg").resolve()),
        "budget_frontier_png": str((artifact_root / "figure_budget_frontier.png").resolve()),
        "budget_frontier_pdf": str((artifact_root / "figure_budget_frontier.pdf").resolve()),
        "budget_frontier_svg": str((artifact_root / "figure_budget_frontier.svg").resolve()),
        "dflash_branch_cache_failure_png": str((artifact_root / "figure_dflash_branch_cache_failure.png").resolve()),
        "dflash_branch_cache_failure_pdf": str((artifact_root / "figure_dflash_branch_cache_failure.pdf").resolve()),
        "dflash_branch_cache_failure_svg": str((artifact_root / "figure_dflash_branch_cache_failure.svg").resolve()),
    }
    _write_text(artifact_root / "figure_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
