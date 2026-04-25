# 18-789 Final Project Report

Course final project report for 18-789 (CMU): *Diffusion Drafters for
Speculative Speculative Decoding*.

## Dependencies

You need **`pdflatex`** and **`bibtex`**. TikZ/PGF (used in the system
diagram) ships with a normal TeX Live install.

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y \
  texlive-latex-recommended \
  texlive-latex-extra \
  texlive-fonts-recommended
```

If anything is still missing, install `texlive-full` or add the package
named in the error.

## Build

```bash
make report     # -> final_report.pdf
make clean      # remove LaTeX temps and PDF
```

The figures referenced by `final_report.tex`
(`figure_normalized_speedup.png`, `figure_budget_frontier.png`) live
next to the `.tex` file so the build has no external paths; they are
produced by the evaluation harness in the sibling code repository.

## Authors

Aditya Ramesh, Arav Tewari, Xinyu Jiang.
