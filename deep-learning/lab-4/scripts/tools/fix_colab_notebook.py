"""Repair a notebook fetched with `colab download`, which writes .ipynb files as a Python dict repr instead of JSON.

Usage: python scripts/tools/fix_colab_notebook.py notebooks/<name>.ipynb [...]
Files that already parse as JSON are left untouched. The repaired notebook is validated with nbformat when available.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def fix(path: Path) -> str:
    text = path.read_text()
    try:
        json.loads(text)
        return "already json"
    except json.JSONDecodeError:
        pass
    nb = ast.literal_eval(text)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    try:
        import nbformat

        nbformat.validate(nbformat.reads(path.read_text(), as_version=4))
        status = "repaired and validated"
    except ImportError:
        status = "repaired (nbformat not installed, not validated)"
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    executed = sum(c.get("execution_count") is not None for c in code)
    errors = sum(any(o.get("output_type") == "error" for o in c["outputs"]) for c in code)
    return f"{status}: {len(code)} code cells, {executed} executed, {errors} with errors"


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(f"{arg}: {fix(path=Path(arg))}")
