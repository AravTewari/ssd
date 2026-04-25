# 18-789 final report — see README.md for TeX dependencies.
#   make report  — build final_report.pdf
#   make clean   — remove LaTeX temps and PDF

LATEX := pdflatex
LATEXFLAGS := -interaction=nonstopmode -halt-on-error
DEPS := reportstyle.sty reportstyle.bst algorithm.sty algorithmic.sty fancyhdr.sty

.PHONY: report clean

report: final_report.pdf

final_report.pdf: final_report.tex final_report.bib $(DEPS)
	$(LATEX) $(LATEXFLAGS) final_report.tex
	bibtex final_report
	$(LATEX) $(LATEXFLAGS) final_report.tex
	$(LATEX) $(LATEXFLAGS) final_report.tex

clean:
	rm -f final_report.pdf \
		final_report.aux final_report.bbl final_report.blg final_report.log final_report.out \
		*.toc *.synctex.gz
