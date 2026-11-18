"""Build HW5_315834011.zip at the lab root from notebook/.

Included (same layout as lab-3 / lab-4 plus the template package the notebook imports): the executed
HW5_315834011.ipynb (every code cell must carry outputs and no error), HW5_315834011_report.pdf from the lab root,
src/*.py and requirements.txt. Excluded: the TA notebook, the dataset symlink, checkpoints/, result.csv, __pycache__ and
.ipynb_checkpoints. Pass --allow-unexecuted to package a notebook without outputs (for a dry run).
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = LAB / "notebook"
NOTEBOOK = NOTEBOOK_DIR / "HW5_315834011.ipynb"
REPORT = LAB / "HW5_315834011_report.pdf"
TARGET = LAB / "HW5_315834011.zip"


def check_notebook(path: Path) -> None:
    nb = json.loads(path.read_text())
    code_cells = [cell for cell in nb["cells"] if cell["cell_type"] == "code"]
    unexecuted = [i for i, cell in enumerate(code_cells) if cell.get("execution_count") is None]
    errors = [
        i
        for i, cell in enumerate(code_cells)
        if any(output.get("output_type") == "error" for output in cell.get("outputs", []))
    ]
    if errors:
        raise SystemExit(f"notebook has error outputs in code cells {errors}")
    if unexecuted and "--allow-unexecuted" not in sys.argv:
        raise SystemExit(f"notebook has unexecuted code cells {unexecuted}; pass --allow-unexecuted for a dry run")
    print(f"{path.name}: {len(code_cells)} code cells, {len(unexecuted)} unexecuted, {len(errors)} errors")


def main() -> int:
    check_notebook(NOTEBOOK)
    members = [NOTEBOOK, NOTEBOOK_DIR / "requirements.txt", *sorted((NOTEBOOK_DIR / "src").glob("*.py"))]
    if not REPORT.exists():
        raise SystemExit(f"missing {REPORT}")
    with zipfile.ZipFile(TARGET, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in members:
            archive.write(member, arcname=str(member.relative_to(NOTEBOOK_DIR)))
        archive.write(REPORT, arcname=REPORT.name)
    with zipfile.ZipFile(TARGET) as archive:
        for info in archive.infolist():
            print(f"  {info.file_size:>10,}  {info.filename}")
    print(f"wrote {TARGET} ({TARGET.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
