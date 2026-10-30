"""Execute the HW4 notebook's code cells on a small subset to prove the pipeline runs end to end.

Overrides: 256 train / 32 val images, 1 epoch, 5 test images, outputs to the scratchpad.
Also measures peak GPU memory of the U-Net training step.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

LAB = Path(__file__).resolve().parents[2]
SCRATCH = Path(sys.argv[1])  # output directory for the smoke run
NB = Path(sys.argv[2]) if len(sys.argv) > 2 else LAB / "HW4_315834011.ipynb"

os.chdir(LAB)
nb = json.loads(NB.read_text())
ns: dict = {}
t0 = time.time()
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    # ---- overrides -------------------------------------------------------
    if "OUTPUT_DIR = DATA_ROOT.parent" in src:
        src += "\nOUTPUT_DIR = Path(r'%s') / 'checkpoints'\nOUTPUT_DIR.mkdir(exist_ok=True)\nCSV_DIR = Path(r'%s')\n" % (SCRATCH, SCRATCH)
    if src.startswith("X_train = train_df['id'].to_numpy()"):
        src += "\nX_train = X_train[:256]\nX_val = X_val[:32]\n"
    if "epoch = 20" in src:
        src = src.replace("epoch = 20", "epoch = 1")
    if "for i in tqdm(range(len(test_set)), desc='Predict'):" in src:
        src = src.replace("range(len(test_set))", "range(5)")
    if src.startswith("plot_loss(history)"):
        continue  # plots need a display
    if src.strip() == "model":
        continue
    print(f"--- cell {i} ({time.time() - t0:.0f}s)", flush=True)
    exec(compile(src, f"cell{i}", "exec"), ns)

import torch  # noqa: E402

print("peak GPU memory during run: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
hist = ns["history"]
print("history:", {k: v if k != "lrs" else len(v) for k, v in hist.items()})
csv_path = SCRATCH / "output.csv"
import csv  # noqa: E402

with csv_path.open() as f:
    rows = list(csv.DictReader(f))
pm = json.loads(rows[0]["pred_mask"])
print("csv rows:", len(rows), "| columns:", list(rows[0].keys()), "| mask shape:", len(pm), "x", len(pm[0]),
      "| value range:", min(min(r) for r in pm), max(max(r) for r in pm))
print("csv bytes per row:", csv_path.stat().st_size // len(rows))
print("SMOKE OK")
