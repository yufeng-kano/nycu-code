---
name: report-writing-workflow
description: "Step-by-step process for course lab reports (PDF): XeLaTeX 繁中, layout rules, figures from logs, visual review of every page, honest numbers"
metadata:
  node_type: memory
  type: feedback
  originSessionId: c72091d1-506b-4ba1-8668-e3609b075ecc
  modified: 2026-10-13T19:23:41.765Z
---

Lab reports (e.g. `HW3_<SID>_report.pdf`, page limit from the spec) are written for the TAs, in 繁體中文, concise. User let me choose tooling; this process produced the Lab 3 report (2026-10-13). Reference implementation: `deep-learning/lab-3/report/` (`main.tex`, `make_figures.py`, `confusion_matrix.py`).

Process:
1. **Read the spec first**: page limit, required items (Lab 3: experimental design, comparisons, findings, compare 3 activations), filename.
2. **Figures from data, not by hand**: `make_figures.py` parses logs → vector PDF in `figures/`, generated at final printed width (`TEXT_WIDTH` in inches) so figure text stays ~7–8pt. One extra analysis figure from a checkpoint (e.g. val confusion matrix, on CPU if the GPU is busy) adds a real finding. Follow the dataviz skill; highlight what matters (e.g. color only error cells).
3. **Typesetting**: `latexmk -xelatex`; fonts per [[report-fonts]]; A4, ~2.2cm side margins; 10.5pt on 17.5pt leading; block paragraphs with parskip; `\raggedright` in narrow columns and table cells (justified narrow text leaves big gaps); no math mode, use Unicode (τ, ε, ×, −, `10\textsuperscript{−3}`) so everything stays in SF Pro; declare en dash/minus/arrow/× as xeCJK `Default` char class; `~=~` so "τ =" doesn't break; captions 表/圖 with table captions on top; footer page numbers "n / N" (fancyhdr + lastpage).
4. **Layout**: title + key-number strip (main results at a glance); pair text with its table/figure side by side (minipage) where it saves space; keep each figure on the same page as its discussion; start a major section on a new page rather than orphan its heading; prose for explanations, bullets only for findings.
5. **Placeholders**: unknown values (final notebook val, Kaggle) are macros rendered in red 「待補」; fill them when results arrive, then check every number is consistent across text, tables and figures (e.g. script vs notebook baseline).
6. **Visual review after every compile**: `pdftoppm -r 100` every page and actually look. Check balance (no all-text or all-figure page, no big empty block), clipped or overlapping labels, font consistency, embedded fonts (`pdffonts`). Overlap-free is not enough.
7. **Hand-off**: copy to the required filename, `latexmk -c`, say what changed, list unverified claims.

**Why:** first draft passed my overlap checks but user said "很醜以及 layout 不合理"; 1.2–1.44× leading was called unreadable. Lab 3 numbers also showed the stats lesson below: script 93.25% vs notebook rerun 91.33% (same config), Kaggle 90.0–90.2%.

**How to apply:** follow the steps for any future report. Before claiming small deltas, rerun the final config at least once; treat run-to-run variance (≈2% in Lab 3), not val sampling noise (0.5%), as the significance floor, and say so in the report. Related: [[concise-theory-plus-code]], [[user-identity]].
