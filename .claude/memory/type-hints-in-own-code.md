---
name: type-hints-in-own-code
description: Code I write for the user (incl. filling lab TODOs) must carry type hints
metadata:
  node_type: memory
  type: feedback
  originSessionId: ffdbc35c-a984-4fa3-bfca-2f9843cd5f42
  modified: 2026-09-27T16:08:31.120Z
---

Any code I write for the user must have type hints: new function signatures, and variable annotations (`x: torch.Tensor = ...`, `i: int = 0`, `self.w: torch.Tensor = ...`) for code filled into provided frameworks. Leave the provided framework's own signatures as-is.

**Why:** User said "自己寫的程式碼要有 type hint" while doing NYCU DL Lab 2.
**How to apply:** Annotate everything I add; related: [[concise-theory-plus-code]].
