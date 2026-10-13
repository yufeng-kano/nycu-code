---
name: discuss-design-before-code
description: "For design choices (model architecture, loss, training strategy), present options and let user pick before writing code"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1d899921-af0e-42db-877a-4a3e33deb54b
  modified: 2026-10-07T06:16:57.684Z
---

When a TODO involves a design decision the user hasn't made (e.g. CNN architecture: VGG-style vs residual vs ConvNeXt-like), first lay out the options with trade-offs and a recommendation, and let the user choose; write the code only after.

**Why:** 2026-10-07 DL Lab 3 — I handed over a full ResNet-style model unasked; user said "我沒有說我要用 Residual 架構 你不應該跟我討論嗎".

**How to apply:** short comparison table + recommendation → ask → then code. Mechanical TODOs (train loop, metrics) can still be coded directly per [[concise-theory-plus-code]].
