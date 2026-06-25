# LaTeX Report

Build with XeLaTeX (recommended):

```bash
cd latex
xelatex main.tex
xelatex main.tex   # second pass for TOC + cross-references
```

Or with LuaLaTeX:

```bash
lualatex main.tex
lualatex main.tex
```

Greek fonts: `DejaVu Serif` + `DejaVu Sans Mono`. Most TeX Live / MacTeX
installations ship with these. If `fontspec` complains, install
`texlive-fonts-extra` or switch to alternative fonts in `preamble.sty`.

## Filling in the Phase 6+ numbers

Every Phase 6+ number that hasn't been measured yet is marked as
`\todoMetric{...}` and renders as a red TODO box in the PDF. Search for
the literal string `\todoMetric` to find them:

```bash
grep -rn '\\todoMetric' chapters/
```

Replace each with the actual value from `runs/<run-name>/metrics.csv` or
the test-split evaluator output.
