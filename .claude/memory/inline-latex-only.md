---
name: inline-latex-only
description: "Use inline $...$ math only; $$ display blocks render badly in the user's VS Code chat panel"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1d899921-af0e-42db-877a-4a3e33deb54b
  modified: 2026-10-07T06:17:00.184Z
---

Write formulas with single-dollar inline math (`$y = \sigma(F(x) + x)$`), never `$$...$$` display blocks.

**Why:** user reported "$$ latex 會怪怪的" in the Claude Code VS Code panel (2026-10-07).

**How to apply:** every reply with math; put a long formula on its own line but still wrapped in single `$`. Related: [[concise-theory-plus-code]].
