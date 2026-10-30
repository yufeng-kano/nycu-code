"""Build the report figures from the experiment logs in ../logs and the executed HW4 notebook.

Figures (vector PDF, generated at the printed width so labels stay 7-8 pt):
  figures/curves.pdf     val mIoU and train loss per epoch, one line per run that has a log
  figures/per_class.pdf  per-class IoU of the best run next to each class's share of labelled pixels
Runs without a log are skipped, so the script works while experiments are still running.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAB_DIR: Path = Path(__file__).resolve().parent.parent
LOG_DIR: Path = LAB_DIR / "logs"
FIG_DIR: Path = Path(__file__).resolve().parent / "figures"
# HW4_NOTEBOOK=path overrides the executed notebook, e.g. to build the figures from a downloaded copy before the swap.
NOTEBOOK: Path = Path(os.environ.get("HW4_NOTEBOOK", str(LAB_DIR / "HW4_315834011.ipynb")))

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
    "Background", "Hat", "Hair", "Glove", "Sunglasses", "Upper-clothes", "Dress", "Coat", "Socks", "Pants",
    "Torso-skin", "Scarf", "Skirt", "Face", "Left-arm", "Right-arm", "Left-leg", "Right-leg", "Left-shoe", "Right-shoe",
]
# Share of labelled (non-255) pixels over train + val masks, from scripts/tools/validate_data.py (2026-10-29).
PIXEL_SHARE: list[float] = [
    59.7015, 0.6629, 4.7969, 0.0636, 0.0416, 9.9178, 1.8939, 7.6151, 0.1239, 4.0220,
    1.5143, 0.2039, 0.3429, 4.6349, 1.5964, 1.7252, 0.3601, 0.3574, 0.2120, 0.2136,
]

# (label, kind, source, colour). kind "script" reads logs/<source>.log; kind "notebook" reads the first training loop of
# the executed ipynb and "notebook:1" the second one (the mini-SAM loop of the final notebook).
RUNS: list[tuple[str, str, str, str]] = [
    ("U-Net (notebook)", "notebook", str(NOTEBOOK), INK_SECONDARY),
    ("002 mini-SAM, G4", "script", "002-mini-sam-g4", ORANGE),
    ("003 mini-SAM wide, G4", "script", "003-mini-sam-wide", AQUA),
    ("004 mini-SAM HR, G4", "script", "004-mini-sam-hr", PURPLE),
    ("005 mini-SAM cctt, G4", "script", "005-mini-sam-cctt", GOLD),
    ("009 mini-SAM wide 150 ep, G4", "script", "009-mini-sam-wide-150ep", TEAL),
    ("008 mini-SAM wide deep, G4", "script", "008-mini-sam-wide-deep", PINK),
    ("014 mini-SAM Lovasz, G4", "script", "014-mini-sam-lovasz", BLUE),
    ("011 mini-SAM 448, G4", "script", "011-mini-sam-448", BROWN),
    ("final mini-SAM 448, 150 ep (notebook)", "notebook:1", str(NOTEBOOK), RED),  # skipped until the notebook has run
]
BEST_RUN: str = "notebook:1"  # per-class figure source: the final notebook's mini-SAM loop

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

SCRIPT_EPOCH = re.compile(
    r"^epoch (\d+)/(\d+) \| lr ([\d.e+-]+) \| train loss ([\d.]+) \| "
    r"val raw ([\d.]+)/([\d.]+)/([\d.]+) ema ([\d.]+)/([\d.]+)/([\d.]+)"
)
SCRIPT_PER_CLASS = re.compile(r"^\s+per-class IoU \((raw|ema)\): (.*)$")
NOTEBOOK_EPOCH = re.compile(
    r"Epoch:(\d+)/(\d+)\.\. Train Loss: ([\d.]+)\.\. Val Loss: ([\d.]+)\.\. "
    r"Train mIoU:([\d.]+)\.\. Val mIoU: ([\d.]+)\.\.(?:.*? \| EMA Val mIoU: ([\d.]+))?"
)


def parse_epoch_lines(lines: list[str], split_runs: bool = True) -> list[dict[str, np.ndarray]]:
    """Turn epoch lines of either format into runs.

    Script format: "epoch N/M | lr L | train loss X | val raw a/b/c ema d/e/f ..." followed on record epochs by
    "  per-class IoU (raw|ema): Name=0.93 ..."; template format: "Epoch:N/M.. Train Loss: X.. Val Loss: Y..
    Train mIoU:Z.. Val mIoU: W.. ..." optionally with " | EMA Val mIoU: V ..." further along the line. With split_runs
    an epoch-1 line starts a new run (several training loops in one notebook) unless the current run has the same
    epoch total and is unfinished: then the lines are a replay after a resume and the last occurrence of each epoch
    wins. The curve takes the better of raw and EMA; per_class is the line printed for the best epoch (the saved
    checkpoint), else the last one printed (the final notebook prints it once, after training).
    """
    runs: list[dict] = []

    def record(epoch: int, total: int, train_loss: float, val_raw: float, val_ema: float) -> None:
        replay = bool(runs) and runs[-1]["total"] == total and len(runs[-1]["val_raw"]) < total
        if not runs or (split_runs and epoch == 1 and not replay):
            runs.append({"total": total, "train_loss": [], "val_raw": [], "val_ema": [], "per_class_by_epoch": {}, "last_epoch": 0})
        run = runs[-1]
        index = epoch - 1
        if index < len(run["val_raw"]):  # replayed epoch: the last occurrence wins
            run["train_loss"][index], run["val_raw"][index], run["val_ema"][index] = train_loss, val_raw, val_ema
        else:
            run["train_loss"].append(train_loss)
            run["val_raw"].append(val_raw)
            run["val_ema"].append(val_ema)
        run["last_epoch"] = epoch

    for line in lines:
        match = SCRIPT_EPOCH.match(line)
        if match:
            record(int(match.group(1)), int(match.group(2)), float(match.group(4)), float(match.group(5)), float(match.group(8)))
            continue
        match = NOTEBOOK_EPOCH.search(line)
        if match:
            ema = float(match.group(7)) if match.group(7) else float("nan")
            record(int(match.group(1)), int(match.group(2)), float(match.group(3)), float(match.group(6)), ema)
            continue
        match = SCRIPT_PER_CLASS.match(line)
        if match and runs:  # names are truncated in the log, so use class order
            runs[-1]["per_class_by_epoch"][runs[-1]["last_epoch"]] = [
                float(tok.split("=")[1]) for tok in match.group(2).split()
            ]
    out = []
    for run in runs:
        raw = np.array(run["val_raw"])
        ema = np.array(run["val_ema"])
        best = np.where(np.isnan(ema), raw, np.maximum(raw, np.nan_to_num(ema, nan=0.0)))
        by_epoch = run["per_class_by_epoch"]
        per_class = by_epoch.get(int(np.argmax(best)) + 1, by_epoch[max(by_epoch)] if by_epoch else [])
        out.append({
            "train_loss": np.array(run["train_loss"]),
            "val_miou": best,
            "val_raw": raw,
            "val_ema": ema,
            "per_class": np.array(per_class),
        })
    return out


def parse_script_log(name: str) -> dict[str, np.ndarray] | None:
    path = LOG_DIR / f"{name}.log"
    if not path.is_file():
        return None
    runs = parse_epoch_lines(lines=path.read_text().splitlines(), split_runs=False)  # resumed logs stay one run
    return runs[0] if runs else None


def parse_notebook(path: str, run_index: int = 0) -> dict[str, np.ndarray] | None:
    """Run number run_index (0 = first training loop in the notebook, 1 = second, ...)."""
    nb_path = Path(path)
    if not nb_path.is_file():
        return None
    text = "".join(
        "".join(out.get("text", "")) if isinstance(out.get("text", ""), list) else out.get("text", "")
        for cell in json.loads(nb_path.read_text())["cells"]
        if cell["cell_type"] == "code"
        for out in cell.get("outputs", [])
    )
    runs = parse_epoch_lines(lines=text.splitlines())
    return runs[run_index] if run_index < len(runs) else None


def load_runs() -> list[tuple[str, dict[str, np.ndarray], str]]:
    loaded = []
    for label, kind, source, colour in RUNS:
        if kind.startswith("notebook"):
            data = parse_notebook(path=source, run_index=int(kind.partition(":")[2] or 0))
        else:
            data = parse_script_log(name=source)
        if data is not None:
            loaded.append((label, data, colour))
    return loaded


def figure_curves(runs: list[tuple[str, dict[str, np.ndarray], str]]) -> None:
    fig, (ax_miou, ax_loss) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(TEXT_WIDTH, 2.2),
        gridspec_kw={"width_ratios": [1.6, 1]},
    )
    for label, data, colour in runs:
        epochs = np.arange(1, len(data["val_miou"]) + 1)
        ax_miou.plot(
            epochs,
            data["val_miou"] * 100,
            color=colour,
            linewidth=1.5,
            label=f"{label}: {data['val_miou'].max() * 100:.1f}",
        )
        ax_loss.plot(np.arange(1, len(data["train_loss"]) + 1), data["train_loss"], color=colour, linewidth=1.5)
    ax_miou.set_xlabel("epoch")
    ax_miou.set_ylabel("val mIoU (%), best of raw / EMA")
    ax_miou.set_title("(a) validation mIoU per epoch", fontsize=8, loc="left", color=INK)
    ax_miou.legend(loc="lower right", fontsize=6.5, labelspacing=0.25, handlelength=1.6)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train loss")
    ax_loss.set_title("(b) training loss", fontsize=8, loc="left", color=INK)
    fig.tight_layout(pad=0.3, w_pad=1.5)
    fig.savefig(FIG_DIR / "curves.pdf")
    plt.close(fig)


def figure_per_class(run: dict[str, np.ndarray], label: str) -> None:
    iou = run["per_class"] * 100
    order = np.arange(len(CLASS_NAMES))
    share = np.array(PIXEL_SHARE)
    fig, (ax_iou, ax_share) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(TEXT_WIDTH, 2.2),
        gridspec_kw={"width_ratios": [1.5, 1]},
        sharey=True,
    )
    colours = [ORANGE if s < 0.5 else BLUE for s in share]
    ax_iou.barh(order, iou[order], color=colours, height=0.7)
    for y, v in zip(order, iou[order]):
        ax_iou.text(x=v + 0.8, y=y, s=f"{v:.0f}", va="center", fontsize=6, color=INK)
    ax_iou.set_yticks(ticks=order, labels=CLASS_NAMES, fontsize=5.5)  # 20 rows in about 1.5 in: 5.2 pt per row
    ax_iou.tick_params(axis="y", pad=1.5, length=0)
    ax_iou.margins(y=0.02)
    ax_iou.invert_yaxis()
    ax_iou.set_xlim(0, 100)
    ax_iou.set_xlabel(f"val IoU (%), {label}", labelpad=1.5)
    ax_iou.set_title("(a) per-class IoU", fontsize=8, loc="left", color=INK, pad=3)
    ax_iou.grid(visible=False, axis="y")
    ax_share.barh(order, share[order], color=colours, height=0.7)
    ax_share.set_xscale("log")
    ax_share.set_xlim(0.03, 100)
    ax_share.set_xlabel("share of labelled pixels (%), log scale", labelpad=1.5)
    ax_share.set_title("(b) class frequency", fontsize=8, loc="left", color=INK, pad=3)
    ax_share.grid(visible=False, axis="y")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(FIG_DIR / "per_class.pdf")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("no run with results found")
    figure_curves(runs=runs)
    print("curves.pdf:", ", ".join(f"{label} ({len(data['val_miou'])} epochs, best {data['val_miou'].max():.4f})" for label, data, _ in runs))
    if BEST_RUN.startswith("notebook"):  # "notebook" / "notebook:1": a training loop printed in the HW4 notebook
        best = parse_notebook(path=str(NOTEBOOK), run_index=int(BEST_RUN.partition(":")[2] or 0))
    else:
        best = parse_script_log(name=BEST_RUN)
    if best is None or best["per_class"].size != len(CLASS_NAMES):
        fallback = [(label, data) for label, data, _ in runs if data["per_class"].size == len(CLASS_NAMES)]
        if not fallback:
            print("per_class.pdf skipped: no run with a per-class line yet")
            return
        label, best = fallback[-1]
    else:  # the legend label of the matching RUNS entry (kind for notebook runs, log name for script runs)
        labels = {kind if kind.startswith("notebook") else source: label for label, kind, source, _ in RUNS}
        label = labels.get(BEST_RUN, BEST_RUN)
    figure_per_class(run=best, label=label)
    print("per_class.pdf from", label)


if __name__ == "__main__":
    main()
