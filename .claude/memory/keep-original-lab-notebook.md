---
name: keep-original-lab-notebook
description: Keep the course-provided template .ipynb untouched and committed; do work in a renamed copy
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1d899921-af0e-42db-877a-4a3e33deb54b
  modified: 2026-10-07T05:39:07.683Z
---

Never `git mv` / rename the TA-provided template notebook. Keep it pristine and tracked in git; make a copy named per submission rules (e.g. `HW3_315834011.ipynb`) and do all work there.

**Why:** user wants the original assignment notebook preserved for reference alongside their solution.

**How to apply:** at the start of each lab, `cp` the template to the submission filename and commit both; if the template gets modified (e.g. IDE re-saves outputs into it), restore it from git. See [[user-identity]].
