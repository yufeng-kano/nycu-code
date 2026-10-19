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
- Edit code and docs in `~/repo/nycu-code` on idk-server; execute notebooks in the Jupyter container `jupyter-code-server` on the same host. Its `/jupyter` is a bind mount of `~/service/jupyter/notebooks`, so `/jupyter/nycu-code` is a second, root-owned checkout (host path `~/service/jupyter/notebooks/nycu-code`), not the host repo.
- Keep the container checkout on the same branch and `git pull` it before running. The container has no GitHub push credentials: bring its commits to the host with `docker exec jupyter-code-server git -C /jupyter/nycu-code bundle create /tmp/x.bundle origin/<branch>..<branch>`, `docker cp jupyter-code-server:/tmp/x.bundle .`, then `git fetch x.bundle <branch>` and push from the host.
- Re-execute from the host (the container venv has ipykernel but not nbconvert, so it is added per run):

  ```sh
  docker exec -w /jupyter/nycu-code/data-science-software-and-computation-experiments/hw2 jupyter-code-server \
    uv run --project /app --no-sync --with nbconvert \
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 315834011_HW2.ipynb
  ```

  Then check that no cell has an error output (`/app/.venv/bin/python` inside the container can parse the ipynb; there is no `python3` on its PATH).
