---
name: report-fonts
description: Licensed Apple fonts live in /jupyter/fonts/; reports use PingFang TC + SF Pro with CJK-friendly spacing
metadata:
  node_type: memory
  type: user
  originSessionId: c72091d1-506b-4ba1-8668-e3609b075ecc
  modified: 2026-10-13T19:13:26.295Z
---

User owns licensed Apple fonts, archived at `/jupyter/fonts/` in folders 中文 (PingFang, Songti, Kaiti), 英文無襯線 (SF Pro Text/Display, Helvetica Neue), 英文襯線 (New York, Charter, Iowan), 等寬 (SF Mono, Menlo). Registered with fontconfig via symlink `/usr/local/share/fonts/apple -> /jupyter/fonts` + `fc-cache -f`. Don't question the license again; the user confirmed it.

Default for all reports from now on (user: "之後就用該字體"):
- Body: PingFang TC Regular + SF Pro Text; bold: Semibold. Headings: SF Pro Display Semibold.
- Figures (matplotlib): "SF Pro Text"; clear `~/.cache/matplotlib` and `/jupyter/.cache/matplotlib` after font changes.
- Spacing: CJK body ~10.5pt on ~17.5pt leading (≈1.65×); the 1.2–1.44× default was called unreadable. CJK–Latin glue ≈0.25em.

Related: [[report-writing-workflow]].
