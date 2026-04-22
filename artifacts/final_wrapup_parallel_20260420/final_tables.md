# Diffusion-as-SSD Final Wrap-Up

## Table 1: Setup
| field | value |
| --- | --- |
| target | `/raid/catalyst/models/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218` |
| DFlash/DDTree draft | `/raid/catalyst/models/hub/models--z-lab--Qwen3-8B-DFlash-b16/snapshots/5adfac36a234741b344cd906ca3fc6a94d7d5955` |
| DFlash predictor | `/raid/user_data/aravt/ssd/artifacts/dummy_dflash_predictor` |
| hardware | `2x B200` |
| held-out prompts | `78` prompt groups from `/raid/user_data/aravt/ssd/artifacts/ddtree_prompt_split/training_metadata.json` |
| main output length | `32` |
| extra output length | `128` |
| AR draft candidates used | `qwen4b=/raid/catalyst/models/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554` |
| AR draft candidates missing | `none` |

## Table 2: Main Decomposition Matrix
| family | mode | b | tok/s | accepted suffix | cache hit | 95% CI tok/s | 95% CI acc | setting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AR | exact_off | 1 | 40.06 | 2.534 | 0.0 | [43.25, 49.24] | [2.583, 2.913] | k=4, fanout_template_name=baseline48 |
| AR | exact_off | 2 | 72.66 | 2.527 | 0.0 | [42.54, 48.42] | [2.576, 2.912] | k=4, fanout_template_name=baseline48 |
| AR | exact_off | 4 | 125.52 | 2.535 | 0.0 | [41.89, 47.44] | [2.584, 2.914] | k=4, fanout_template_name=baseline48 |
| DFLASH | exact_off | 1 | 62.48 | 2.739 | 0.0 | [86.84, 114.66] | [3.086, 4.045] | fanout_template_name=baseline48 |
| DFLASH | exact_off | 2 | 95.56 | 2.739 | 0.0 | [72.67, 96.27] | [3.065, 4.036] | fanout_template_name=baseline48 |
| DFLASH | exact_off | 4 | 129.99 | 2.739 | 0.0 | [58.38, 77.87] | [3.062, 4.029] | fanout_template_name=baseline48 |
| DDTREE | exact_off | 1 | 82.04 | 3.082 | 0.0 | [95.63, 115.17] | [3.202, 3.842] | tree_budget=8, frontier_count=2 |
| DDTREE | exact_off | 2 | 125.03 | 3.082 | 0.0 | [74.50, 89.09] | [3.202, 3.840] | tree_budget=8, frontier_count=1 |
| DDTREE | exact_off | 4 | 157.80 | 3.082 | 0.0 | [48.14, 58.34] | [3.199, 3.838] | tree_budget=8, frontier_count=1 |
| AR | exact_on_oracle | 1 | 79.91 | 3.072 | 0.9091981132075472 | [97.98, 116.74] | [3.351, 4.032] | k=8, fanout_template_name=baseline48 |
| AR | exact_on_oracle | 2 | 201.49 | 2.532 | 0.9239130434782609 | [116.73, 131.78] | [2.583, 2.912] | k=4, fanout_template_name=baseline48 |
| AR | exact_on_oracle | 4 | 242.07 | 2.874 | 0.9150110375275938 | [90.31, 104.94] | [3.043, 3.558] | k=6, fanout_template_name=baseline48 |
| DFLASH | exact_on_oracle | 1 | 50.86 | 2.739 | 0.9209445585215605 | [71.36, 94.09] | [3.079, 4.042] | fanout_template_name=baseline48 |
| DFLASH | exact_on_oracle | 2 | 91.81 | 2.739 | 0.9209445585215605 | [68.93, 91.05] | [3.063, 4.039] | fanout_template_name=baseline48 |
| DFLASH | exact_on_oracle | 4 | 119.26 | 2.739 | 0.9209445585215605 | [49.47, 64.08] | [3.078, 4.048] | fanout_template_name=baseline48 |
| DDTREE | exact_on_oracle | 1 | 81.45 | 3.451 | 0.9 | [96.66, 119.44] | [3.688, 4.630] | tree_budget=16, frontier_count=1 |
| DDTREE | exact_on_oracle | 2 | 119.31 | 3.082 | 0.9094117647058824 | [70.83, 85.27] | [3.205, 3.845] | tree_budget=8, frontier_count=2 |
| DDTREE | exact_on_oracle | 4 | 141.08 | 3.451 | 0.9 | [44.10, 53.99] | [3.687, 4.621] | tree_budget=16, frontier_count=1 |
| AR | realized | 1 | 63.59 | 2.961 | 0.5947488584474886 | [84.10, 103.47] | [3.181, 3.763] | k=6, fanout_template_name=baseline48 |
| AR | realized | 2 | 154.35 | 2.688 | 0.6270833333333333 | [95.19, 113.86] | [2.782, 3.185] | k=4, fanout_template_name=baseline48 |
| AR | realized | 4 | 125.42 | 3.140 | 0.5698795180722892 | [52.77, 64.11] | [3.464, 4.161] | k=8, fanout_template_name=baseline48 |
| DFLASH | realized | 1 | 24.10 | 2.633 | 0.09090909090909091 | [30.50, 39.97] | [2.957, 3.849] | fanout_template_name=baseline48 |
| DFLASH | realized | 2 | 30.08 | 2.633 | 0.09090909090909091 | [19.21, 25.14] | [2.937, 3.863] | fanout_template_name=baseline48 |
| DFLASH | realized | 4 | 35.51 | 2.633 | 0.09090909090909091 | [11.32, 14.56] | [2.947, 3.840] | fanout_template_name=baseline48 |
| DDTREE | realized | 1 | 67.93 | 3.079 | 0.0035252643948296123 | [78.57, 94.25] | [3.200, 3.839] | tree_budget=8, frontier_count=1 |
| DDTREE | realized | 2 | 97.15 | 3.446 | 0.0038910505836575876 | [58.79, 73.50] | [3.675, 4.630] | tree_budget=16, frontier_count=1 |
| DDTREE | realized | 4 | 131.42 | 3.079 | 0.0035252643948296123 | [39.30, 47.38] | [3.196, 3.836] | tree_budget=8, frontier_count=1 |

## Figure Captions
- `figure_oracle_ceiling.png`: best oracle throughput by batch size for AR, DFlash, and DDTree.
- `figure_normalized_speedup.png`: oracle throughput normalized by each family's exact-off throughput.
- `figure_budget_frontier.png`: oracle throughput vs accepted suffix over swept AR and DDTree budgets, with DFlash fixed points.
- `figure_error_bars.png`: 95% bootstrap confidence intervals for oracle throughput and accepted suffix.

## Appendix Table: Standalone vs Exact-Off
### DFlash
| b | standalone tok/s | exact-off tok/s | delta |
| --- | --- | --- | --- |
| 1 | 62.64 | 62.48 | -0.3% |
| 2 | 99.00 | 95.56 | -3.5% |
| 4 | 143.76 | 129.99 | -9.6% |

### DDTree
| b | N | F | standalone tok/s | exact-off tok/s | delta |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 1 | 14.60 | 16.99 | 16.4% |
| 1 | 8 | 2 | 84.72 | 82.04 | -3.2% |
| 1 | 16 | 1 | 85.86 | 80.18 | -6.6% |
| 1 | 16 | 2 | 66.33 | 63.62 | -4.1% |
| 2 | 8 | 1 | 107.52 | 125.03 | 16.3% |
| 2 | 8 | 2 | 124.40 | 114.79 | -7.7% |
| 2 | 16 | 1 | 114.12 | 107.22 | -6.0% |
| 2 | 16 | 2 | 115.71 | 120.91 | 4.5% |
| 4 | 8 | 1 | 172.89 | 157.80 | -8.7% |
| 4 | 8 | 2 | 158.91 | 15.83 | -90.0% |
| 4 | 16 | 1 | 155.77 | 144.46 | -7.3% |
| 4 | 16 | 2 | 8.93 | 8.18 | -8.4% |

## Appendix DFlash Diagnostics
See `appendix_dflash_diagnostics.md` for the full diagnostic matrix and branch-template sweep.

## Appendix Extra Regime
| family | b | exact-off tok/s | exact-on-oracle tok/s | oracle delta | exact-off acc | oracle acc |
| --- | --- | --- | --- | --- | --- | --- |
| AR | 1 | 101.86 | 78.70 | -22.7% | 1.929 | 2.063 |
| AR | 2 | 154.42 | 183.00 | 18.5% | 2.067 | 1.924 |
| AR | 4 | 283.91 | 327.61 | 15.4% | 2.072 | 1.924 |
| DFLASH | 1 | 85.48 | 81.21 | -5.0% | 3.256 | 3.256 |
| DFLASH | 2 | 138.61 | 129.55 | -6.5% | 3.256 | 3.256 |
| DFLASH | 4 | 225.73 | 193.67 | -14.2% | 3.256 | 3.256 |
| DDTREE | 1 | 105.22 | 106.85 | 1.6% | 4.035 | 4.035 |
| DDTREE | 2 | 157.08 | 156.18 | -0.6% | 3.610 | 3.610 |
| DDTREE | 4 | 197.62 | 218.99 | 10.8% | 4.035 | 3.610 |

## Appendix Dream Note
- This document summarizes the first end-to-end benchmark results for using a synchronous Dream diffusion model as the draft model in this speculative decoding engine.
- | Dream diffusion | `b=8`, `dsteps=8` | `57.72 tok/s` |
- | Dream diffusion | `b=8`, `dsteps=16` | `37.30 tok/s` |
- | AR baseline | `b=8`, `Qwen3-8B` draft | `342.48 tok/s` |
- - Against the best Dream point we tested, the autoregressive baseline is about `5.9x` faster.
- - Against Dream at `dsteps=16`, the autoregressive baseline is about `9.2x` faster.

## Final Recommendation
```json
{
  "extra_regime_self_speedups": [
    {
      "batch_size": 1,
      "family": "DFLASH",
      "oracle_vs_exact_off_delta_frac": -0.18599099726250135,
      "regime": "main"
    },
    {
      "batch_size": 2,
      "family": "DFLASH",
      "oracle_vs_exact_off_delta_frac": -0.039201034840251975,
      "regime": "main"
    },
    {
      "batch_size": 4,
      "family": "DFLASH",
      "oracle_vs_exact_off_delta_frac": -0.0825048880264064,
      "regime": "main"
    },
    {
      "batch_size": 1,
      "family": "DFLASH",
      "oracle_vs_exact_off_delta_frac": -0.04997723377166555,
      "regime": "extra"
    },
    {
      "batch_size": 2,
      "family": "DFLASH",
      "oracle_vs_exact_off_delta_frac": -0.06539427865552704,
      "regime": "extra"
    },
    {
      "batch_size": 4,
      "family": "DFLASH",
      "oracle_vs_exact_off_delta_frac": -0.1420315710943753,
      "regime": "extra"
    },
    {
      "batch_size": 1,
      "family": "DDTREE",
      "oracle_vs_exact_off_delta_frac": -0.00717288710775415,
      "regime": "main"
    },
    {
      "batch_size": 2,
      "family": "DDTREE",
      "oracle_vs_exact_off_delta_frac": -0.04569755358360084,
      "regime": "main"
    },
    {
      "batch_size": 4,
      "family": "DDTREE",
      "oracle_vs_exact_off_delta_frac": -0.1059720101561684,
      "regime": "main"
    },
    {
      "batch_size": 1,
      "family": "DDTREE",
      "oracle_vs_exact_off_delta_frac": 0.015512809674443432,
      "regime": "extra"
    },
    {
      "batch_size": 2,
      "family": "DDTREE",
      "oracle_vs_exact_off_delta_frac": -0.005685389954423579,
      "regime": "extra"
    },
    {
      "batch_size": 4,
      "family": "DDTREE",
      "oracle_vs_exact_off_delta_frac": 0.10813026441019678,
      "regime": "extra"
    }
  ],
  "message": "At least one diffusion family still shows meaningful headroom under the final wrap-up criteria.",
  "oracle_vs_ar": [
    {
      "ar_oracle_tok_s": 79.9053468450128,
      "batch_size": 1,
      "delta_vs_ar_oracle_frac": -0.36353479693913543,
      "family": "DFLASH",
      "oracle_tok_s": 50.85697280535989
    },
    {
      "ar_oracle_tok_s": 201.49466325555738,
      "batch_size": 2,
      "delta_vs_ar_oracle_frac": -0.5443485514795127,
      "family": "DFLASH",
      "oracle_tok_s": 91.81133518154255
    },
    {
      "ar_oracle_tok_s": 242.06560493571888,
      "batch_size": 4,
      "delta_vs_ar_oracle_frac": -0.5073198047536938,
      "family": "DFLASH",
      "oracle_tok_s": 119.26092950214522
    },
    {
      "ar_oracle_tok_s": 79.9053468450128,
      "batch_size": 1,
      "delta_vs_ar_oracle_frac": 0.01939095636660544,
      "family": "DDTREE",
      "oracle_tok_s": 81.45478793914292
    },
    {
      "ar_oracle_tok_s": 201.49466325555738,
      "batch_size": 2,
      "delta_vs_ar_oracle_frac": -0.4078611792721704,
      "family": "DDTREE",
      "oracle_tok_s": 119.31281228309689
    },
    {
      "ar_oracle_tok_s": 242.06560493571888,
      "batch_size": 4,
      "delta_vs_ar_oracle_frac": -0.4171971683572324,
      "family": "DDTREE",
      "oracle_tok_s": 141.07651999985646
    }
  ],
  "stop_project": false
}
```
