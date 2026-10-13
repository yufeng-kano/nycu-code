---
name: explicit-keyword-args
description: "In code I write, pass every function/constructor argument by keyword name (size=, p=, mean=), not positionally"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1d899921-af0e-42db-877a-4a3e33deb54b
  modified: 2026-10-07T05:53:21.485Z
---

Write calls with explicit keyword argument names, e.g. `transforms.RandomResizedCrop(size=IMG_SIZE, scale=(0.6, 1.0))`, `transforms.RandomHorizontalFlip(p=0.5)`, `transforms.Normalize(mean=MEAN, std=STD)` — including params left at a meaningful default the user should see (like `p=0.5`).

**Why:** user (2026-10-07, DL Lab 3) said arg names must be labeled to be "correct"; positional calls hide what each number means.

**How to apply:** all code for labs, including TODO fills and examples in explanations. Pairs with [[type-hints-in-own-code]] and [[concise-theory-plus-code]].
