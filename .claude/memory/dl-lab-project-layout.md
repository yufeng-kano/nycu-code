---
name: dl-lab-project-layout
description: "DL lab folders must mirror lab-3's layout (scripts/logs/result/checkpoints/report, numbered NNN-name runs, data folder gitignored); user said don't place files arbitrarily"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 64810614-4935-4046-8ce1-5ef8e791df1f
  modified: 2026-10-29T13:26:45.712Z
---

For every DL lab folder, mirror `deep-learning/lab-3/` exactly:

- Lab root: TA template `.ipynb` + `.pdf` untouched, `HWn_SID.ipynb`, `HWn_SID_report.pdf`, `kaggle-link.txt` (invite link), final submission CSV at root (`*.csv` is gitignored), `HWn_SID.zip` (gitignored).
- Data: one folder at lab root named as Kaggle ships it (lab-3: `Lab3_data_intel_2026/`), a real copy, listed explicitly in the repo `.gitignore`.
- Experiments: `scripts/train/NNN-name.py` ↔ `logs/NNN-name.log` ↔ `result/NNN-name.csv` ↔ `checkpoints/NNN-name.pt` (same number + slug; `.pt` gitignored); `result/summary.md` tracks runs. From lab-4 on, `scripts/` is split: `scripts/train/` for numbered experiment runs, `scripts/tools/` for download / validation / inference / packaging helpers (user asked for the split on 2026-10-29 and left the exact names to me).
- Report: `report/main.tex`, `report/make_figures.py`, `report/figures/*.pdf`; final PDF copied to lab root.

**Why:** user (2026-10-29, lab-4 start): "你去看一下 lab 3 的專案格式不要亂擺".

**How to apply:** before creating any file in a new lab, `ls` lab-3 and place the file in the matching spot; never symlink data into the repo (a symlink is not caught by the `**/data/` ignore rule). See [[keep-original-lab-notebook]], [[git-commit-policy]].
