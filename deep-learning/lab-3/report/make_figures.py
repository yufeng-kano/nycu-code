"""Build report figures from the experiment logs in ../logs."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

LOG_DIR: Path = Path(__file__).resolve().parent.parent / "logs"
FIG_DIR: Path = Path(__file__).resolve().parent / "figures"

BLUE: str = "#2a78d6"
ORANGE: str = "#eb6834"
AQUA: str = "#1baf7a"
INK: str = "#0b0b0b"
INK_SECONDARY: str = "#52514e"
GRID: str = "#e4e3df"
TEXT_WIDTH: float = 16.6 / 2.54  # A4 with 2.2 cm side margins, in inches

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

LINE_PATTERN = re.compile(
    r"^\[(SUP|SSL) (\d+)/\d+\].*\| val ([\d.]+)/[\d.]+(?:.*\(pseudo (\d+)\))?"
)


def parse_log(name: str) -> dict[str, np.ndarray]:
    sup_acc: list[float] = []
    ssl_acc: list[float] = []
    pseudo: list[int] = []
    for line in (LOG_DIR / f"{name}.log").read_text().splitlines():
        match = LINE_PATTERN.match(line)
        if match is None:
            continue
        stage, _, acc, n_pseudo = match.groups()
        if stage == "SUP":
            sup_acc.append(float(acc))
        else:
            ssl_acc.append(float(acc))
            pseudo.append(int(n_pseudo))
    return {
        "val_acc": np.array(sup_acc + ssl_acc),
        "pseudo": np.array(pseudo),
        "num_sup": np.array(len(sup_acc)),
    }


# Figure 1: one change at a time, from the notebook baseline to the final setting
steps: list[str] = [
    "baseline",
    "+ sup\n200 ep",
    "+ drop path\n0.2",
    "+ width\n512",
    "+ SSL\n200 ep",
    "+ pseudo-label\nevery 5 ep",
    "+ erasing\np = 0.5",
]
sup_best: list[float] = [0.8842, 0.8883, 0.8992, 0.8967, 0.8967, 0.8967, 0.9042]
ssl_best: list[float] = [0.8958, 0.9025, 0.9100, 0.9117, 0.9175, 0.9183, 0.9325]

fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 1.65))
x: np.ndarray = np.arange(len(steps))
ax.plot(
    x,
    np.array(ssl_best) * 100,
    color=ORANGE,
    linewidth=2,
    marker="o",
    markersize=4,
    label="after self-training",
)
ax.plot(
    x,
    np.array(sup_best) * 100,
    color=BLUE,
    linewidth=2,
    marker="o",
    markersize=4,
    label="supervised only",
)
for xi, yi in zip(x, ssl_best):
    ax.annotate(
        f"{yi * 100:.2f}",
        xy=(xi, yi * 100),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=7,
        color=INK,
    )
ax.set_xticks(ticks=x, labels=steps)
ax.set_ylabel("best val acc (%)")
ax.set_ylim(87.8, 94.4)
ax.grid(visible=False, axis="x")
ax.margins(x=0.05)
ax.legend(loc="upper left")
fig.tight_layout(pad=0.3)
fig.savefig(FIG_DIR / "progression.pdf")
plt.close(fig)


# Figure 2: activation comparison on the final setting
runs: list[tuple[str, str, str]] = [
    ("GELU", "016-erasing-0.5", BLUE),
    ("ReLU", "020-act-relu", ORANGE),
    ("LeakyReLU (0.1)", "021-act-leaky-relu", AQUA),
]
window: int = 9

fig, (ax_acc, ax_pl) = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(TEXT_WIDTH, 3.3),
    gridspec_kw={"width_ratios": [2.2, 1]},
)
for label, name, color in runs:
    log = parse_log(name=name)
    smoothed: np.ndarray = np.convolve(log["val_acc"], np.ones(window) / window, mode="valid")
    epochs: np.ndarray = np.arange(len(smoothed)) + window // 2 + 1
    ax_acc.plot(epochs, smoothed * 100, color=color, linewidth=1.5, label=label)
    ax_pl.plot(
        np.arange(1, len(log["pseudo"]) + 1),
        log["pseudo"],
        color=color,
        linewidth=1.5,
        label=label,
    )

num_sup: int = int(parse_log(name="016-erasing-0.5")["num_sup"])
ax_acc.axvline(x=num_sup + 0.5, color=INK_SECONDARY, linewidth=0.8, linestyle="--")
ax_acc.text(
    x=num_sup - 4,
    y=93.0,
    s="self-training starts",
    ha="right",
    fontsize=7,
    color=INK_SECONDARY,
)
ax_acc.set_xlim(40, 400)
ax_acc.set_ylim(78, 94)
ax_acc.set_yticks(ticks=range(78, 95, 2))
ax_acc.set_xlabel("epoch (supervised 1–200, self-training 201–400)")
ax_acc.set_ylabel(f"val acc (%), {window}-epoch mean")
ax_acc.set_title("(a) validation accuracy", fontsize=8, loc="left", color=INK)
ax_acc.legend(loc="lower center", ncols=3, fontsize=7)

ax_pl.set_xlabel("self-training epoch")
ax_pl.set_ylabel("accepted (of 3600)")
ax_pl.set_ylim(0, 3600)
ax_pl.set_title("(b) pseudo-labels", fontsize=8, loc="left", color=INK)

fig.tight_layout(pad=0.3, w_pad=1.5)
fig.savefig(FIG_DIR / "activation.pdf")
plt.close(fig)


# Figure 3: validation confusion matrix of the best checkpoint (from confusion_matrix.py)
class_names: list[str] = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
confusion: np.ndarray = np.loadtxt(fname=FIG_DIR / "confusion_val.csv", delimiter=",", dtype=np.int64)
error_rate: np.ndarray = confusion / confusion.sum(axis=1, keepdims=True)
np.fill_diagonal(error_rate, np.nan)
cmap = LinearSegmentedColormap.from_list(name="errors", colors=["#ffffff", "#f9d3c2", ORANGE, "#a63c12"])
cmap.set_bad(color="#eef1f5")

fig, ax = plt.subplots(figsize=(0.44 * TEXT_WIDTH, 0.44 * TEXT_WIDTH * 0.95))
ax.imshow(np.ma.masked_invalid(error_rate), cmap=cmap, vmin=0, vmax=0.12)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        if confusion[i, j] == 0:
            continue
        ax.text(
            x=j,
            y=i,
            s=str(confusion[i, j]),
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold" if i == j else "normal",
            color="white" if i != j and error_rate[i, j] > 0.06 else INK,
        )
ax.set_xticks(ticks=range(len(class_names)), labels=class_names, rotation=35, ha="right")
ax.set_yticks(ticks=range(len(class_names)), labels=class_names)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.grid(visible=False)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(FIG_DIR / "confusion.pdf")
plt.close(fig)
