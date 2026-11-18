"""Build the report figures from the experiment logs in ../logs and the executed HW5 notebook.

Figures (vector PDF, generated at the printed width so labels stay 7-8 pt):
  figures/curves.pdf     val AP and training loss per epoch, one line per run that has a log
  figures/per_class.pdf  per-class AP of the submitted model next to each class's share of small boxes
Runs without a log are skipped, so the script works while experiments are still running.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAB_DIR: Path = Path(__file__).resolve().parent.parent
LOG_DIR: Path = LAB_DIR / "logs"
FIG_DIR: Path = Path(__file__).resolve().parent / "figures"
NOTEBOOK: Path = LAB_DIR / "notebook" / "HW5_315834011.ipynb"

BLUE: str = "#2a78d6"
ORANGE: str = "#eb6834"
AQUA: str = "#1baf7a"
PURPLE: str = "#7d5bd6"
GOLD: str = "#c9a200"
TEAL: str = "#2aa198"
PINK: str = "#d6409f"
BROWN: str = "#8a6d3b"
RED: str = "#d03b3b"
INK: str = "#0b0b0b"
INK_SECONDARY: str = "#52514e"
GRID: str = "#e4e3df"
TEXT_WIDTH: float = 16.6 / 2.54  # A4 with 2.2 cm side margins, in inches

CLASS_NAMES: list[str] = [
    "chair", "couch", "dining table", "tv", "laptop", "mouse", "keyboard", "remote", "cell phone", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "book", "clock", "vase", "potted plant",
]
# Train boxes per class (crowd excluded) and the share of them with area < 32^2 (COCO "small"), from train.txt (2026-11-07).
TRAIN_COUNTS: list[int] = [
    9116, 1391, 3759, 1377, 1132, 529, 669, 1384, 1483, 5762,
    1914, 4892, 1309, 1817, 1397, 3447, 5646, 1554, 1515, 1895,
]
SMALL_SHARE: list[float] = [
    0.398, 0.014, 0.153, 0.089, 0.140, 0.560, 0.164, 0.743, 0.624, 0.608,
    0.502, 0.484, 0.578, 0.660, 0.685, 0.324, 0.697, 0.463, 0.445, 0.294,
]

# (label, source, colour): source is a log name under logs/ or "notebook" for the executed HW5 notebook.
RUNS: list[tuple[str, str, str]] = [
    ("001 YOLOv3 template", "001-yolov3-baseline", INK_SECONDARY),
    ("002 anchor-free, ConvNeXt-T, 640", "002-tal-convnext-t", ORANGE),
    ("005 60 epoch", "005-short-schedule", GOLD),
    ("006 800 letterbox", "006-size-800", TEAL),
    ("009 = 006 with Darknet-53", "009-darknet53", BROWN),
    ("007 ConvNeXt-S", "007-convnext-s", PINK),
    ("008 multi-scale up to 1000", "008-scale-1000", BLUE),
    ("011 ConvNeXt-B, 40 epoch", "011-convnext-b-40ep", "#1f5f8b"),
    ("013 neck 384 / depth 3", "013-neck-384", "#6b8e23"),
    ("015 + copy-paste", "015-copy-paste", "#444e9c"),
    ("final: train + val, hold-out AP", "final-notebook", RED),
]
BEST_RUN: str = "notebook"  # per-class figure source

plt.rcParams.update(
    {
        "font.family": "SF Pro Text",
        "font.size": 8,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK_SECONDARY,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    }
)

EPOCH = re.compile(
    r"^epoch (\d+)/(\d+) \| (?:train )?loss ([\d.]+).*?\| ([\d.]+) min"
    r"(?: \| AP ([\d.]+) \| AP50 ([\d.]+) \| AP75 ([\d.]+) \| APs ([\d.]+) \| APm ([\d.]+) \| APl ([\d.]+))?"
)
PER_CLASS = re.compile(r"^per-class AP(?: \(last eval\))?: (.*)$")


def parse_lines(lines: list[str]) -> dict:
    """Epoch lines -> {total, epochs, loss, ap_epochs, ap, ap_small, ap_large, per_class}; the last per-class line wins."""
    run: dict = {"total": 0, "epochs": [], "loss": [], "ap_epochs": [], "ap": [], "ap_small": [], "ap_large": [], "per_class": {}}
    for line in lines:
        match = EPOCH.match(line)
        if match:
            epoch, total, loss = int(match.group(1)), int(match.group(2)), float(match.group(3))
            if epoch <= len(run["epochs"]):  # replay after a resume: the last occurrence wins
                run["epochs"] = run["epochs"][: epoch - 1]
                run["loss"] = run["loss"][: epoch - 1]
                keep = [i for i, e in enumerate(run["ap_epochs"]) if e < epoch]
                for key in ("ap_epochs", "ap", "ap_small", "ap_large"):
                    run[key] = [run[key][i] for i in keep]
            run["total"] = total
            run["epochs"].append(epoch)
            run["loss"].append(loss)
            if match.group(5):
                run["ap_epochs"].append(epoch)
                run["ap"].append(float(match.group(5)))
                run["ap_small"].append(float(match.group(8)))
                run["ap_large"].append(float(match.group(10)))
            continue
        match = PER_CLASS.match(line)
        if match:
            pairs = [item.rsplit(" ", 1) for item in match.group(1).split(", ")]
            run["per_class"] = {name: float(value) for name, value in pairs}
    return run


def notebook_lines(path: Path) -> list[str]:
    nb = json.loads(path.read_text())
    lines: list[str] = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                text = output.get("text", "")
                lines.extend("".join(text).splitlines())
    return lines


def load_run(source: str) -> dict | None:
    if source == "notebook":
        if not NOTEBOOK.exists():
            return None
        lines = notebook_lines(NOTEBOOK)
    else:
        path = LOG_DIR / f"{source}.log"
        if not path.exists():
            return None
        lines = path.read_text().splitlines()
    run = parse_lines(lines)
    return run if run["ap"] else None


def curves(runs: list[tuple[str, dict, str]]) -> None:
    figure, (ax_ap, ax_loss) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.0), gridspec_kw={"width_ratios": [1.3, 1]})
    for label, run, colour in runs:
        width = 1.6 if colour == RED else 1.0
        ax_ap.plot(run["ap_epochs"], run["ap"], color=colour, linewidth=width, marker="o", markersize=2.2, label=label)
        if run["loss"] and not label.startswith("001"):
            ax_loss.plot(run["epochs"], run["loss"], color=colour, linewidth=width, label=label)
    ax_ap.set_xlabel("epoch")
    ax_ap.set_ylabel("val AP (final: hold-out), EMA weights")
    ax_ap.set_ylim(0.0, 0.50)
    ax_ap.set_xlim(0, 75)
    ax_ap.set_title("(a) validation AP@[0.50:0.95] per epoch", loc="left", fontsize=8, color=INK)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("training loss (box + cls + DFL)")
    ax_loss.set_xlim(0, 75)
    ax_loss.set_ylim(2.5, 9.0)
    ax_loss.set_title("(b) training loss (001 omitted, different loss)", loc="left", fontsize=8, color=INK)
    handles, labels = ax_ap.get_legend_handles_labels()
    ax_loss.legend(handles, labels, fontsize=5.5, loc="upper right", handlelength=1.4, labelspacing=0.15, borderaxespad=0.2)
    figure.tight_layout(w_pad=1.5)
    figure.savefig(FIG_DIR / "curves.pdf")
    plt.close(figure)


def per_class(run: dict) -> None:
    names = [name for name in CLASS_NAMES if name in run["per_class"]]
    values = np.array([run["per_class"][name] for name in names])
    small = np.array([SMALL_SHARE[CLASS_NAMES.index(name)] for name in names])
    order = np.argsort(values)[::-1]
    names = [names[i] for i in order]
    values, small = values[order], small[order]
    colours = [ORANGE if s >= 0.5 else BLUE for s in small]

    figure, (ax_ap, ax_small) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.2), gridspec_kw={"width_ratios": [1, 1]})
    y = np.arange(len(names))
    ax_ap.barh(y, values, color=colours, height=0.65)
    ax_ap.set_yticks(y, names, fontsize=6)
    ax_ap.invert_yaxis()
    ax_ap.set_xlim(0, 0.90)
    ax_ap.set_xlabel("hold-out AP of the final model, with TTA")
    for yi, v in zip(y, values):
        ax_ap.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=5.5, color=INK_SECONDARY)
    ax_ap.grid(axis="y", visible=False)
    ax_ap.set_title("(a) per-class AP", loc="left", fontsize=8, color=INK)

    ax_small.barh(y, small * 100, color=colours, height=0.7)
    ax_small.set_yticks(y, names, fontsize=6)
    ax_small.invert_yaxis()
    ax_small.set_xlim(0, 80)
    ax_small.set_xlabel("share of train boxes with area < 32² px (%)")
    ax_small.grid(axis="y", visible=False)
    ax_small.set_title("(b) small-object share (orange: at least 50%)", loc="left", fontsize=8, color=INK)
    figure.tight_layout(w_pad=2.0)
    figure.savefig(FIG_DIR / "per_class.pdf")
    plt.close(figure)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    loaded: list[tuple[str, dict, str]] = []
    for label, source, colour in RUNS:
        run = load_run(source)
        if run is None:
            print(f"skip {label}: no log yet")
            continue
        best = int(np.argmax(run["ap"]))
        print(f"{label}: best AP {run['ap'][best]:.4f} at epoch {run['ap_epochs'][best]} of {run['total']}")
        loaded.append((label, run, colour))
    curves(loaded)
    best_run = load_run(BEST_RUN)
    if best_run and best_run["per_class"]:
        per_class(best_run)
        ranked = sorted(best_run["per_class"].items(), key=lambda item: item[1])
        print("weakest:", ", ".join(f"{k} {v:.3f}" for k, v in ranked[:5]))
        print("strongest:", ", ".join(f"{k} {v:.3f}" for k, v in ranked[-4:]))
    print(f"figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
