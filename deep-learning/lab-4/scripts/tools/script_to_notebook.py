"""Turn an experiment script into a notebook whose cells are the script's sections.

Usage:
    python scripts/tools/script_to_notebook.py scripts/train/002-mini-sam-g4.py notebooks/002-mini-sam-g4.ipynb

The banner comments (`# ===== #` / `# <Section> #` / `# ===== #`) delimit the cells, so the notebook holds
exactly the code of the script. The `if __name__ == "__main__":` block becomes a run cell that mirrors it
(log file + Tee + main) without argparse. When that cell resumes from a `-last.pt` checkpoint it first
replays the existing log, so its output is the complete training record even after a Colab session loss.
Execute with `jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 <nb>` to record
every cell's output.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BANNER = re.compile(r"^# =+ #\n#\s*(?P<title>.*?)\s*#?\n# =+ #\n", re.MULTILINE)


def split_sections(source: str) -> list[tuple[str, str]]:
    """Return (title, code) pairs; the text before the first banner is the preamble."""
    matches = list(BANNER.finditer(source))
    assert matches, "no section banners found"
    sections: list[tuple[str, str]] = [("Preamble", source[: matches[0].start()])]
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(source)
        sections.append((m.group("title"), source[m.end() : end]))
    return sections


def code_cell(lines: list[str]) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def markdown_cell(lines: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def main(script: Path, out: Path) -> int:
    source = script.read_text()
    head, _, tail = source.partition('\nif __name__ == "__main__":\n')
    assert tail, "no __main__ block found"
    sections = split_sections(head)
    name = re.search(r'^EXPERIMENT_NAME: str = "(.+?)"', source, re.MULTILINE).group(1)

    cells: list[dict] = [
        markdown_cell([
            f"# {name}\n",
            "\n",
            f"Generated from `{script.as_posix()}` by `scripts/tools/script_to_notebook.py`: each cell below is one ",
            "section of that script, executed in order with `jupyter nbconvert --execute`, so the outputs are the ",
            "actual training record of this run.",
        ]),
        code_cell([
            "import os\n",
            "from pathlib import Path\n",
            "\n",
            'os.environ.setdefault("LAB_DIR", str(Path.cwd()))  # the script derives LAB_DIR from __file__, which a notebook lacks\n',
            'print("LAB_DIR =", os.environ["LAB_DIR"])\n',
        ]),
    ]
    for title, code in sections:
        code = code.strip("\n") + "\n"
        if not code.strip():
            continue
        cells.append(markdown_cell([f"## {title}\n"]))
        cells.append(code_cell(code.splitlines(keepends=True)))

    cells.append(markdown_cell(["## Run\n"]))
    cells.append(code_cell([
        "resuming = LAST_PATH.is_file()\n",
        'assert resuming or not LOG_PATH.exists(), f"{LOG_PATH} already exists, refuse to overwrite"\n',
        "LOG_PATH.parent.mkdir(parents=True, exist_ok=True)\n",
        "if resuming:  # replay the log of the earlier part of this run so the record below is complete\n",
        '    print(LOG_PATH.read_text(), end="")\n',
        'log_file = LOG_PATH.open("a" if resuming else "w")\n',
        "_stdout, _stderr = sys.stdout, sys.stderr\n",
        "sys.stdout, sys.stderr = Tee(_stdout, log_file), Tee(_stderr, log_file)\n",
        "try:\n",
        "    main(CONFIG, smoke=False)\n",
        "finally:\n",
        "    sys.stdout, sys.stderr = _stdout, _stderr\n",
        "    log_file.close()\n",
    ]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out} with {len(cells)} cells: " + ", ".join(t for t, _ in sections))
    return 0


if __name__ == "__main__":
    sys.exit(main(script=Path(sys.argv[1]), out=Path(sys.argv[2])))
