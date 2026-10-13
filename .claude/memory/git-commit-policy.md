---
name: git-commit-policy
description: What goes into the nycu-code git repo — report PDFs and memory yes, model weights no
metadata:
  type: feedback
---

In `/jupyter/nycu-code` (remote `yufeng-kano/nycu-code`, **public**):
- Commit the final report PDF (e.g. `HW3_<SID>_report.pdf`) and its LaTeX/figure sources.
- Never commit model weights (`*.pt`, checkpoints); `.gitignore` already has `*.pt`. Datasets and `*.csv` are ignored too.
- Claude memory lives in the repo at `.claude/memory/` and is committed with the work; `/root/.claude/projects/-jupyter-nycu-code/memory` is a symlink to it.

**Why:** user said (2026-10-13) "memory 要進入 git ... pdf 也要記住權重不用".

**How to apply:** when committing, include memory changes and report PDFs, check `git status` for stray weights. Because the repo is public and course rules forbid sharing solutions/predictions before the deadline, confirm with the user before pushing graded work ahead of its due date. Related: [[report-writing-workflow]].
