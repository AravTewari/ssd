# Extra SSD-Favorable Regime

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
