"""Generate side-by-side per-family oracle-vs-realized plot.

Addresses presentation feedback:
"It would be better to show the oracle and the realized performance on
the same plot. You could have 3 side-by-side plots, one for each draft
family, and in each plot you show the two curves comparing oracle to
realized."
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

SRC = Path(
    "/sgl-workspace/dgm/ssd/artifacts/final_wrapup_parallel_20260420/final_summary.json"
)
OUT = Path(
    "/sgl-workspace/dgm/ssd/report/18789_final_report/figures/fig09b_oracle_vs_realized_per_family.png"
)

with open(SRC) as f:
    summary = json.load(f)

rows = summary["main_decomposition_rows"]


def collect(family: str, mode: str):
    sel = [r for r in rows if r["family"] == family and r["mode"] == mode]
    sel.sort(key=lambda r: r["batch_size"])
    bs = [r["batch_size"] for r in sel]
    tps = [r["throughput_tok_s"] for r in sel]
    lo = [r["confidence_intervals"]["throughput_tok_s"]["low"] for r in sel]
    hi = [r["confidence_intervals"]["throughput_tok_s"]["high"] for r in sel]
    err_low = [max(t - l, 0.0) for t, l in zip(tps, lo)]
    err_high = [max(h - t, 0.0) for t, h in zip(tps, hi)]
    return bs, tps, [err_low, err_high]


FAMILIES = [
    ("AR", "Autoregressive (Qwen3-0.6B)"),
    ("DFLASH", "DFlash (one-pass diffusion)"),
    ("DDTREE", "DDTree (tree diffusion)"),
]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)

for ax, (fam_key, fam_label) in zip(axes, FAMILIES):
    bs_off, tps_off, err_off = collect(fam_key, "exact_off")
    bs_or, tps_or, err_or = collect(fam_key, "exact_on_oracle")
    bs_re, tps_re, err_re = collect(fam_key, "realized")

    ax.errorbar(
        bs_off, tps_off, yerr=err_off,
        marker="s", linestyle=":", color="#7f8c8d",
        capsize=3, label="Exact-off (SD)", linewidth=1.6,
    )
    ax.errorbar(
        bs_or, tps_or, yerr=err_or,
        marker="o", linestyle="-", color="#2980b9",
        capsize=3, label="Oracle (SSD ceiling)", linewidth=2.0,
    )
    ax.errorbar(
        bs_re, tps_re, yerr=err_re,
        marker="^", linestyle="--", color="#c0392b",
        capsize=3, label="Realized (SSD)", linewidth=2.0,
    )

    ax.set_title(fam_label, fontsize=11)
    ax.set_xlabel("Batch size $b$", fontsize=10)
    ax.set_xticks([1, 2, 4])
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

axes[0].set_ylabel("Decode throughput (tok/s)", fontsize=10)
axes[0].legend(loc="upper left", fontsize=8.5, frameon=True)

fig.suptitle(
    "Oracle ceiling vs. realized SSD throughput, per drafter family "
    "(Qwen3-8B verifier, output_len=32, greedy decode, 2$\\times$B200)",
    fontsize=11, y=1.02,
)

fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved: {OUT}")
