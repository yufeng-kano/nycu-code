---
name: per-folder-gitignore
description: Ignore rules for one assignment go in that assignment folder's own .gitignore, not the repo root
metadata:
  type: feedback
---

In `/jupyter/nycu-code`, ignore rules that concern a single assignment folder go in a `.gitignore` inside that folder (paths relative to it). The root `.gitignore` is for repo-wide patterns only. A nested `!data/` re-includes a folder that the root `**/data/` rule ignores.

**Why:** DS HW3 (2026-11-18): I added hw3-only rules to the root file; the user said 「你不應該是在我們這個資料夾單獨用一個 ignore 嗎」. Other folders already do this (`iot/hw4-linebot/.gitignore`, `hw2-docker-compose/.gitignore`).

**How to apply:** when an untracked local file should stay out of git, add it to the folder-level `.gitignore`; verify with `git check-ignore`. Related: [[git-commit-policy]].
