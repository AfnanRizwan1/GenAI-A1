# Technical report (IEEE format, LaTeX)

* `main.tex`, `references.bib`: the paper (IEEEtran conference class).
* `generated/`: LaTeX tables written by `scripts/make_report_assets.py` from the real experiment outputs.
* `figures/`: result figures (copied / plotted by the same script) and the Stitch design screenshots.

Build after the experiments have finished:

```bash
python scripts/make_report_assets.py --outputs outputs --models models --out report
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

or upload the whole `report/` folder to Overleaf (main file `main.tex`). Tables and figures that are not available yet show a
"pending" placeholder, so the document always compiles. Text marked `TODO` in red still needs the final numbers or your own input.
