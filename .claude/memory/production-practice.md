---
name: production-practice
description: For infra/deployment homework, default to production-grade practice over the assignment's letter, and always state where the setup still differs from production
metadata:
  type: feedback
---

When an assignment's suggested setup differs from standard production practice, pick the production practice when it is cheap, and list every remaining deviation explicitly (what we do, what production does, why).

**Why:** HW3 (2026-10-19): user said 「如果 server 不跑 Pod 是正規 production 作法的話就這樣做，你也是有不正規的地方你要說，不然就是誤人子弟」, and 「不管題目 pgdb 就用 statefulset」 even though the assignment asked for a Deployment.

**How to apply:** when proposing a design, say which choice is the production norm and recommend it; flag where the assignment's wording conflicts so the user can accept the grading risk. Keep a "與 production 的差異" list in the docs. Still discuss design before code ([[discuss-design-before-code]]).
