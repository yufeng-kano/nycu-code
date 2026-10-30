---
name: dl-lab4-iteration-loop
description: "Standing instruction for DL Lab 4 (given 2026-10-29): iterate models on Colab G4 continuously until the user says stop, cost-effective changes only, search papers when stuck, no token waste on log syncing"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 73580a1a-7579-4baf-aaa8-2c931cf7c7ee
  modified: 2026-10-29T16:11:18.890Z
---

For DL Lab 4 the user said (2026-10-29): keep iterating models on the two Colab G4 sessions until they say stop; update `result/summary.md` after every finished run; prefer cost-effective architecture changes (no complexity for a small gain); when progress stalls, search papers/docs for architecture improvements; keep at least one request per hour so the prompt cache stays warm (session cron tick); do not spend turns manually syncing logs (background pollers do it). Added 2026-10-29 23:40: no more Kaggle submissions until the user says so ("不要提交，就一直改進"); the public leaderboard (top 0.726, we were 5th at 0.640) is read with `kaggle competitions leaderboard --show`. Added 2026-10-29 23:55: end state is a single `HW4_315834011.ipynb` carrying the best recipe (training + inference chain + CSV); the user will run it themselves once on Colab end to end and submit the CSV by hand ("最後整合到 HW4_315834011.ipynb 我之後要自己去 colab 跑一次完整 train 生成出 csv 並手動提交"), so the final recipe must be one run that fits one Colab session and must be resumable. Added 2026-10-30 00:05: one model only, no self-pretraining stage, no ensemble ("不要雙模型 ... 也不能用 pre-train"); SimMIM runs were cancelled. Status 2026-10-30 04:50: the loop ended on budget grounds (both Colab sessions stopped, ~140 units left, the user's own final run needs ~80); final recipe = 011 (008 model at 448) with 150 epochs, inference sizes 512-768 + hflip + min-area 50; `HW4_315834011_minisam.ipynb` built and smoke-tested, waiting for the user to run it on Colab and to approve the swap. Do not start new Colab runs unless the user asks.

**Why:** the user wants the best Kaggle score with minimal supervision and minimal token cost; they were annoyed by turn-by-turn log moving and script running without visible purpose.

**How to apply:** after each poller completion: record the run in `result/summary.md` and `CLAUDE.md`, pick the next experiment (one clear change per run, ablation against the current best), launch it on the freed session, tell the fork session the new log name, and go quiet until the next event. Report only at launch, failure and completion. See [[dl-lab4-cihp]] and [[discuss-design-before-code]] (the user delegated the architecture choices for this loop, so no per-run design discussion is needed).

Status 2026-10-30 14:05 UTC: the user's own final run is done and verified (val 0.627 raw / 0.628 EMA, hflip TTA 0.633, upload chain 0.6535; `HW4_315834011_final.ipynb`, `checkpoints/minisam-final-best.pt`, `result/minisam-final.csv`), the Colab session stopped on the user's instruction (56 units left). Remaining: the user's hand submission to Kaggle, their OK to swap the executed notebook over `HW4_315834011.ipynb`, and the report (fork session) on that run. Do not start Colab runs.
Update 14:15 UTC: notebook swapped in as `HW4_315834011.ipynb` with the user's OK and the CSV submitted (public 0.65835, submission 57807731). Lab 4 is delivered except the report's Kaggle tile (fork session) and the user's git commit.
