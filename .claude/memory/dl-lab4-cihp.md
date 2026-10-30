---
name: dl-lab4-cihp
description: "DL Lab 4 (CIHP semantic segmentation, due 2026-11-01): status, rules and data-download workaround live in deep-learning/lab-4/CLAUDE.md; read it first"
metadata:
  type: project
---

DL Lab 4 status, decisions, environment notes and the Kaggle data-download workaround are kept in `deep-learning/lab-4/CLAUDE.md`. Read it before doing anything in lab-4.

Key facts as of 2026-10-29: deadline 2026-11-01 23:59; grading 70% per-image mIoU on Kaggle + 30% report (max 3 pages); TA supplement (2026-10-23) forbids any external pretrained weights or data but frees architecture, loss, training procedure and code outside TODO blocks. The Kaggle API cannot serve train/val files for this competition; data came from per-folder zips (website folder download via signed storage links, and a private packing kernel `sakurabaemaouo/lab4-cihp-pack`).

See [[dl-lab-project-layout]]; [[lab3-goal-max-accuracy]] records the priority stated for lab 3, not yet confirmed for lab 4.
