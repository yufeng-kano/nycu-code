---
name: call-formatting-style
description: "User's call layout — 3+ args one per line with trailing comma; 1–2 short args inline"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1d899921-af0e-42db-877a-4a3e33deb54b
  modified: 2026-10-07T06:02:17.690Z
---

User's own style (seen in DL Lab 3 transforms, 2026-10-07):
- Calls with 3+ arguments: break after `(`, one `kw=value,` per line, trailing comma, closing `)` on its own line.
- Calls with 1–2 short arguments stay inline: `RandomHorizontalFlip(p=0.5)`, `Normalize(mean=MEAN, std=STD)`.
- Template lines left by TAs are not reformatted.

**Why:** user asked "你有看到我的書寫習慣嗎" after I handed over everything as one-liners.

**How to apply:** format all code I write this way. Always add the trailing comma on the last arg; the formatter is ruff, and its magic trailing comma is what keeps a call multi-line, without it ruff collapses it back to one line. Combine with [[explicit-keyword-args]].
