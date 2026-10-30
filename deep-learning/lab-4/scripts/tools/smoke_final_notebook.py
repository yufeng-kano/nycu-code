"""Execute the final notebook's code cells on a small subset to prove the whole pipeline runs end to end.

Usage: python scripts/tools/smoke_final_notebook.py OUT_DIR NOTEBOOK

Overrides: 256 train / 32 val images and 1 epoch for the U-Net; 64 images, 2 epochs and 4 workers for the mini-SAM;
5 test images for the CSV; checkpoints and CSV go to OUT_DIR. The mini-SAM training cell is run a second time to
exercise the resume path. Plots are skipped (no display).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

csv.field_size_limit(sys.maxsize)  # each pred_mask field is a 320 x 320 JSON list (about 200 kB)

import matplotlib

matplotlib.use("Agg")

LAB = Path(__file__).resolve().parents[2]
SCRATCH = Path(sys.argv[1]).resolve()
NB = Path(sys.argv[2]).resolve()
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ["LAB4_RUN_DIR"] = str(SCRATCH / "runs")
os.chdir(LAB)

nb = json.loads(NB.read_text())
ns: dict = {}
t0 = time.time()
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "OUTPUT_DIR = DATA_ROOT.parent" in src:
        src += f"\nOUTPUT_DIR = Path(r'{SCRATCH}') / 'checkpoints'\nOUTPUT_DIR.mkdir(exist_ok=True)\nCSV_DIR = Path(r'{SCRATCH}')\n"
    if src.startswith("X_train = train_df['id'].to_numpy()"):
        src += "\nX_train = X_train[:256]\nX_val = X_val[:32]\n"
    if "epoch = 20" in src:
        src = src.replace("epoch = 20", "epoch = 1")
    if "CONFIG = ExperimentConfig()" in src:
        src += "\nimport dataclasses\nCONFIG = dataclasses.replace(CONFIG, epochs=2, warmup_epochs=1, num_workers=4)\nSMOKE_SUBSET = 64\n"
    if "for i in tqdm(range(len(test_set)), desc='Predict'):" in src:
        src = src.replace("range(len(test_set))", "range(5)")
    if src.startswith("plot_loss(") or src.strip() == "model":
        continue
    print(f"--- cell {i} ({time.time() - t0:.0f}s)", flush=True)
    exec(compile(src, f"cell{i}", "exec"), ns)
    if src.startswith("history_sam = train_minisam"):
        print("--- resume check: running the training cell again", flush=True)
        exec(compile(src, f"cell{i}-again", "exec"), ns)

import torch  # noqa: E402

print("peak GPU memory during run: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
print("history_sam:", {k: [round(x, 4) for x in v] for k, v in ns["history_sam"].items()})
assert (SCRATCH / "runs" / "minisam-final-best.pt").is_file() and (SCRATCH / "runs" / "minisam-final-last.pt").is_file()
csv_path = SCRATCH / "output.csv"
with csv_path.open() as f:
    rows = list(csv.DictReader(f))
pm = json.loads(rows[0]["pred_mask"])
print("csv rows:", len(rows), "| columns:", list(rows[0].keys()), "| mask shape:", len(pm), "x", len(pm[0]),
      "| value range:", min(min(r) for r in pm), max(max(r) for r in pm))
assert len(rows) == 5 and len(pm) == 320 and len(pm[0]) == 320
print("SMOKE OK")
