# CLAUDE.md: DS HW2 ATP tennis analysis

Notes for Claude working in this folder. The spec is `AGENTS.md`; the deliverable is `315834011_HW2.ipynb` only. Status and decisions go here.

## Status (2026-10-19)

- **Done** parts 1–3 and all 5 custom questions (height distribution, win correlates, left-handers, surfaces, ranking gap and upsets). The whole notebook executes cleanly with outputs saved.
- **Submit** the notebook file itself; nothing else is required.

## Decisions

- Match files follow the English spec, `atp_matches_202[1-4].csv` (main tour only, 11,712 matches); the Chinese wording would also pull in qualifiers and challengers. The notebook says so.
- Win rates per player use a 50-match floor (per-surface rankings use 20, because grass has few events). Walkovers (`score == "W/O"`) are excluded from surface stats.
- Ranking gap is measured as a ratio of the two ranks, not a rank difference, consistent with the log10 rank in question 2.

## Conventions

- Markdown cells are Traditional Chinese, flat `##` plus bold question lines, every term explained from scratch (repo memory `document-readability`). Figures use the palette defined in the first code cell.
- Code is fully typed, keyword arguments everywhere, 3+ arguments one per line with a trailing comma.

## Running

- Data: `git clone --depth 1 https://github.com/Haz22537/JEFFSM_tennis_atp tennis_atp` inside this folder (git-ignored); the first code cell clones it when missing.
- Re-execute with `jupyter nbconvert --to notebook --execute --inplace 315834011_HW2.ipynb` and check no cell has an error output.
