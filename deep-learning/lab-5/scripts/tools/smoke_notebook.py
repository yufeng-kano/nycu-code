"""Execute a copy of the built notebook end to end: the template YOLOv3 part for 1 epoch, the final detector on a
few images for 2 short epochs.

Usage: smoke_notebook.py OUT.ipynb [NOTEBOOK]
The copy sets `smoke = True` and `num_epochs = 1`, redirects every checkpoint and CSV to notebook/checkpoints/smoke/,
runs with the notebook folder as working directory and is written (with outputs) to OUT.ipynb. Each patch whose
old text is present must match exactly one cell; the `smoke` patch must be present.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

LAB = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = LAB / "notebook"

PATCHES: tuple[tuple[str, str], ...] = (
    ("smoke = False", "smoke = True"),
    ("num_epochs = 50", "num_epochs = 1"),  # template YOLOv3 part
    ('det_checkpoint_dir = Path("checkpoints")', 'det_checkpoint_dir = Path("checkpoints/smoke")'),  # final detector, before the template line it contains
    ('\ncheckpoint_dir = Path("checkpoints")', '\ncheckpoint_dir = Path("checkpoints/smoke/yolo")'),  # template
    ('    output_path=Path("result.csv"),', '    output_path=Path("checkpoints/smoke/result_yolo.csv"),'),  # template
    ('RESULT_PATH = Path("result.csv")', 'RESULT_PATH = Path("checkpoints/smoke/result.csv")'),  # final detector
)


def main() -> int:
    out = Path(sys.argv[1])
    source = Path(sys.argv[2]) if len(sys.argv) > 2 else NOTEBOOK_DIR / "HW5_315834011.ipynb"
    nb = nbformat.read(source, as_version=4)
    for old, new in PATCHES:
        hits = [cell for cell in nb.cells if cell.cell_type == "code" and old in cell.source]
        assert len(hits) <= 1, (old, len(hits))
        if hits:
            hits[0].source = hits[0].source.replace(old, new)
        else:
            assert old != "smoke = False", "the notebook has no smoke flag"
            print(f"patch not applicable: {old!r}")

    client = NotebookClient(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK_DIR)}})
    start = time.time()
    client.execute()
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    n_code = sum(cell.cell_type == "code" for cell in nb.cells)
    errors = [
        output
        for cell in nb.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    print(f"executed {n_code} code cells in {(time.time() - start) / 60:.1f} min, {len(errors)} errors -> {out}")
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        text = "".join(output.get("text", "") for output in cell.get("outputs", []) if output.get("output_type") == "stream")
        lines = [line for line in text.splitlines() if line.startswith(("Epoch", "epoch", "done", "hold-out @", "Wrote", "best ", "Colab:", "Train", "Validation"))]
        for line in lines:
            print("  " + line)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
