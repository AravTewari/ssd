# Figure Index for Final Report

This folder contains every image either extracted from `18789_ Final Project Presentation.pdf` or pulled from the `/sgl-workspace/dgm/ssd` codebase, renamed by content.

## Source tags

- **[Library]**: pulled directly from the `/sgl-workspace/dgm/ssd` repository (high-resolution originals)
- **[PDF]**: extracted from the presentation slides (the repository does not ship a standalone file)
- **[Generated]**: regenerated for the final report from `final_summary.json` via matplotlib (or, for `fig01`, from a hand-authored SVG)

---

## 1. Concept and methodology

| File | Content | Source slide | Origin |
|---|---|---|---|
| `fig01_async_ssd_runtime.png` | **Figure 1 in the report**: Async SSD runtime with a pluggable speculator backend---two lanes (async pre-speculation / synchronous verifier), a Cache Lookup hex, hit/miss branches, and the Accepted tokens sink; the `> this work` chip is placed only over the speculator backend | --- | [Generated] `_fig01_source.svg` (rendered with cairosvg to a 3600x1840 PNG) |
| `fig01_ssd_concept_diagram.png` | SD vs. SSD concept diagram (Speculate / Verify / Predict & Speculate) plus throughput comparison with vLLM/SGLang (4x) | Slide 2 (Motivation) | [Library] `assets/ssd fig1 readme.png` (kept for slide reference; **no longer used in the report**) |

---

## 2. Baseline experiments (Qwen3-32B verifier)

| File | Content | Source slide | Origin |
|---|---|---|---|
| `fig02_baseline_throughput_qwen32b_drafters.png` | Decode throughput by draft model on the Qwen3-32B verifier; AR / SD / SSD comparison across drafters in {0.6B, 1.7B, 4B} | Slide 4 | [PDF] |
| `fig03_baseline_acceptance_cache_metrics.png` | Speculative-decoding metrics by draft model on Qwen3-32B: accept rate / avg tokens per step / SSD cache-hit rate (three-panel) | Slide 5 | [PDF] |
| `fig04a_pareto_frontier_paper_reference.png` | Throughput-latency Pareto frontier (paper reference: Llama-3.1-Instruct 70B / 1B) | Slide 6 (left) | [PDF] |
| `fig04b_pareto_frontier_qwen32b_ours.png` | Throughput-latency Pareto frontier on Qwen3-32B / 0.6B (our reproduction) | Slide 6 (right) | [PDF] |
| `fig05_decode_throughput_vs_speculation_length_k.png` | Decode throughput vs. k on Qwen3-32B, b=1, k in {1, 2, 4, 6, 8, 16} | Slide 7 | [PDF] |

---

## 3. Main results: oracle ceiling and headroom (Qwen3-8B target, 2x H100)

| File | Content | Source slide | Origin |
|---|---|---|---|
| `fig08_oracle_ceiling_comparison_b1to32.png` | **Oracle ceiling comparison**: AR vs. DFlash vs. DDTree, output_len in {32, 128}, b in {1, 2, 4, 8, 16, 32} (with error bars) | Slide 11 | [Library] `modal_figures/figure_oracle_ceiling.png` |
| `fig08b_oracle_ceiling_comparison_b1to4.png` | Same as above but only b in {1, 2, 4} (the main regime used in the paper) | --- | [Library] `figure_oracle_ceiling.png` |
| `fig09_realized_system_throughput_outlen32.png` | Realized system throughput at output length 32; AR / DFlash / DDTree as batch size varies | Slide 12 | [PDF] |
| `fig09b_oracle_vs_realized_per_family.png` | **Per-family** oracle vs. realized vs. exact-off SD throughput; one panel per drafter family with oracle and realized curves on the same axes; 95% bootstrap CIs; generated specifically for the final report (responds to the presentation feedback) | --- | [Generated] `gen_oracle_vs_realized.py` |
| `fig10_normalized_ssd_headroom_b1to32.png` | **Normalized SSD headroom** = oracle tok/s / exact-off tok/s, output_len in {32, 128} | Slide 13 | [Library] `modal_figures/figure_normalized_speedup.png` |
| `fig10b_normalized_ssd_headroom_b1to4.png` | Same as above for b in {1, 2, 4} | --- | [Library] `figure_normalized_speedup.png` |

---

## 4. Realized-system throughput and full Pareto

| File | Content | Source slide | Origin |
|---|---|---|---|
| `fig11_budget_frontier_pareto_b1to32.png` | Throughput vs. accepted-suffix frontier (exact-on-oracle, output_len = 32), b in {1, 2, 4, 8, 16, 32}; six-panel Pareto plot | --- | [Library] `modal_figures/figure_budget_frontier.png` |
| `fig11b_budget_frontier_pareto_b1to4.png` | Same as above but three panels for b in {1, 2, 4} | --- | [Library] `figure_budget_frontier.png` |
| `fig12_throughput_comparison_all_drafters_b1.png` | Throughput by draft model at b=1, covering AR (Qwen3 0.6B / 1.7B / 4B) plus DFlash and DDTree | Slide 15 | [PDF] |
| `fig15_throughput_latency_pareto_full_qwen0p6b.png` | Full throughput-latency Pareto frontier (AR Qwen0.6B / SD / SSD plus DFlash SD/SSD plus DDTree SD/SSD) | Slide 16 | [PDF] |

---

## 5. DFlash diagnostics and training

| File | Content | Source slide | Origin |
|---|---|---|---|
| `fig13_dflash_branch_cache_failure_b1to32.png` | **Appendix: DFlash branch-cache failure**: throughput bars plus cache-hit overlay; under Pred+On+Normal the cache-hit rate stays near 0.1; b in {1, 2, 4, 8, 16, 32} | --- | [Library] `modal_figures/figure_dflash_branch_cache_failure.png` |
| `fig13b_dflash_branch_cache_failure_b1to4.png` | Same as above for b in {1, 2, 4} | --- | [Library] `figure_dflash_branch_cache_failure.png` |
| `fig16_dflash_supported_models_table.png` | "Supported models" table from the DFlash repo, with Qwen3-8B (non-thinking) highlighted | Slide 17 | [PDF] |

---

## 6. Supplementary

| File | Content | Source slide | Origin |
|---|---|---|---|
| `fig14_supplementary_error_bars.png` | Two-panel error-bar scatter (AR / DFlash / DDTree) used for uncertainty quantification | --- | [Library] `artifacts/.../figure_error_bars.png` |
| `fig17_corollary8_strictly_faster_than_sd.png` | Screenshot of "Corollary 8 (Strictly Faster Than SD)"---the prior claim that SSD should never be slower than SD | Slide 19 (Lessons) | [PDF] |
| `fig18_twitter_discussion_tidar_relation.png` | Twitter exchange between Avner May and Aditya Ramesh about the relationship to NVIDIA TiDAR; useful for related-work framing | Slide 20 (Lessons) | [PDF] |

---

## Recommended figures by report section

- **Abstract / Introduction**: `fig01_async_ssd_runtime.png`
- **Related work**: `fig18_twitter_discussion_tidar_relation.png` (TiDAR / "Your LLM Knows the Future" relationship)
- **Methodology**: `fig01_async_ssd_runtime.png`, `fig17_corollary8_strictly_faster_than_sd.png`
- **Baselines (Qwen3-32B)**:
  - `fig02_baseline_throughput_qwen32b_drafters.png`
  - `fig03_baseline_acceptance_cache_metrics.png`
  - `fig04a/b_pareto_frontier_*.png`
  - `fig05_decode_throughput_vs_speculation_length_k.png`
- **Main results (Qwen3-8B oracle / realized)**:
  - `fig08_oracle_ceiling_comparison_b1to32.png` (main figure)
  - `fig09_realized_system_throughput_outlen32.png`
  - `fig10_normalized_ssd_headroom_b1to32.png`
  - `fig11_budget_frontier_pareto_b1to32.png`
  - `fig12_throughput_comparison_all_drafters_b1.png`
  - `fig15_throughput_latency_pareto_full_qwen0p6b.png`
- **DFlash failure analysis**:
  - `fig13_dflash_branch_cache_failure_b1to32.png`
  - `fig16_dflash_supported_models_table.png`
- **Appendix**:
  - `fig14_supplementary_error_bars.png`
  - `fig08b/10b/11b/13b_*_b1to4.png` (the small-batch versions)
