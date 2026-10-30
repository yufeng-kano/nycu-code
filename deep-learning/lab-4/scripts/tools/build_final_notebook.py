"""Build the final HW4 notebook: the TA template with every TODO filled (through build_notebook.py) plus the
mini-SAM final model, whose cells are the sections of a run script so the notebook holds exactly the trained code.

Usage:
    python scripts/tools/build_final_notebook.py scripts/train/011-mini-sam-448.py HW4_315834011_minisam.ipynb \
        [--set epochs=150] [--set img_size=448] [--infer-sizes 320,384,448,512] [--epoch-min 2.8] [--force]

Result: template cells unchanged up to the U-Net plots (data, U-Net, its training run), a Colab data cell after
"Unzip Data", the mini-SAM section (recipe, data, model, loss, optimisation, resumable training, curves, inference
chain) before "Predict Result", and the prediction cells repointed at the mini-SAM checkpoint.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1]
sys.path.insert(0, str(HERE))
from script_to_notebook import split_sections  # noqa: E402


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


def markdown(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def find(cells: list[dict], needle: str) -> int:
    hits = [i for i, c in enumerate(cells) if needle in "".join(c["source"])]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def find_start(cells: list[dict], prefix: str) -> int:
    hits = [i for i, c in enumerate(cells) if "".join(c["source"]).startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def section(sections: dict[str, str], title: str) -> str:
    assert title in sections, (title, list(sections))
    return sections[title].strip("\n") + "\n"


def set_default(src: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(    {re.escape(key)}: [^=\n]+= )[^#\n]*(#.*)?$", re.M)
    assert len(pattern.findall(src)) == 1, key
    return pattern.sub(lambda m: f"{m.group(1)}{value}  {m.group(2) or ''}".rstrip(), src)


DATA_CELL = r'''
# Colab only: mount Google Drive, unpack the data zips kept there and keep checkpoints there (a no-op elsewhere).
import importlib.util
import os
import shutil
import zipfile
from pathlib import Path

ON_COLAB = importlib.util.find_spec("google.colab") is not None
DRIVE_DIR = Path("/content/drive/MyDrive/lab-4")  # DRIVE_DIR/data holds train.zip, train_mask.zip, val.zip, val_mask.zip, test.zip, test_index.csv
EXPECTED_COUNTS = {"train": 28280, "train_mask": 28280, "val": 1000, "val_mask": 1000, "test": 3961}

if ON_COLAB:
    if not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive")
    if importlib.util.find_spec("albumentations") is None:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "albumentations"], check=True)
    data = Path.cwd() / "data"
    for folder, count in EXPECTED_COUNTS.items():
        target = data / folder
        if target.is_dir() and sum(1 for _ in target.glob("*.png")) == count:
            continue
        archive = DRIVE_DIR / "data" / f"{folder}.zip"
        if not archive.is_file():
            raise FileNotFoundError(f"{archive} is missing: download {folder}.zip from the Kaggle Data page into {archive.parent}")
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if member.is_dir() or not member.filename.endswith(".png"):
                    continue
                member.filename = Path(member.filename).name  # the zips may carry a leading folder; the data layout is flat
                z.extract(member, target)
        n = sum(1 for _ in target.glob("*.png"))
        print(f"{folder}: {n} PNGs extracted from {archive.name}")
        assert n == count, f"{folder}: expected {count} PNGs, found {n}"
    if not (data / "test_index.csv").is_file():
        shutil.copy(DRIVE_DIR / "data" / "test_index.csv", data / "test_index.csv")
    runs = DRIVE_DIR / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    os.environ["LAB4_RUN_DIR"] = str(runs)  # the mini-SAM section writes its resumable checkpoint there
    print("Data ready in", data, "| checkpoints go to", runs)
else:
    print("Not on Colab: the local data/ folder and checkpoints/ are used")
'''

PATHS_CELL = r'''
DATA_DIR: Path = DATA_ROOT
RUN_DIR: Path = Path(os.environ.get("LAB4_RUN_DIR", str(OUTPUT_DIR)))  # on Colab this is on Google Drive, so a reclaimed session resumes
RUN_DIR.mkdir(parents=True, exist_ok=True)
LAST_PATH: Path = RUN_DIR / f"{EXPERIMENT_NAME}-last.pt"  # full training state, rewritten after every epoch
CKPT_PATH: Path = RUN_DIR / f"{EXPERIMENT_NAME}-best.pt"  # weights of the best epoch (by val mIoU), used for the CSV
print("checkpoints:", RUN_DIR)
'''

TRAIN_CELL = r'''
def save_state(path: Path, model: nn.Module, ema: EMA, optimizer: optim.Optimizer, **extra) -> None:
    """Everything needed to continue training after a reclaimed Colab session."""
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(), **extra}, tmp)
    tmp.replace(path)


def train_minisam(cfg: ExperimentConfig) -> dict[str, list[float]]:
    """Train the mini-SAM with the recipe above. Resumes from LAST_PATH when it exists (and replays the earlier epoch
    lines so this cell's output stays the complete record); prints one line per epoch in the template's format."""
    start = time.time()
    set_seed(cfg.seed)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Config: {asdict(cfg)}")
    print(f"Device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

    train_ids = list_ids(folder=DATA_DIR / "train")[:SMOKE_SUBSET]
    val_ids = list_ids(folder=DATA_DIR / "val")[:SMOKE_SUBSET]
    train_set = CIHPSegDataset(
        image_dir=DATA_DIR / "train",
        mask_dir=DATA_DIR / "train_mask",
        ids=train_ids,
        transform=build_train_transform(cfg=cfg),
        hflip_prob=0.5,
        size=cfg.img_size,
        resize_mask=True,
    )
    val_set = CIHPSegDataset(
        image_dir=DATA_DIR / "val",
        mask_dir=DATA_DIR / "val_mask",
        ids=val_ids,
        transform=None,
        hflip_prob=0.0,
        size=cfg.img_size,
        resize_mask=False,
    )
    sam_train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    sam_val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    print(f"train {len(train_set)} / val {len(val_set)} images, {len(sam_train_loader)} steps per epoch")

    model = MiniSAM(cfg=cfg).to(device)
    print(f"MiniSAM params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    optimizer = optim.AdamW(params=param_groups(model=model, weight_decay=cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.999))
    ema = EMA(model=model, decay=cfg.ema_decay)
    total_steps = cfg.epochs * len(sam_train_loader)
    warmup_steps = cfg.warmup_epochs * len(sam_train_loader)

    keys = ("train_loss", "val_loss", "train_miou", "val_miou", "train_acc", "val_acc", "lrs", "ema_val_miou")
    history: dict[str, list[float]] = {k: [] for k in keys}
    log_lines: list[str] = []
    best_miou, best_tag, best_per_class = 0.0, "", None
    best_state: dict[str, torch.Tensor] | None = None
    step, start_epoch = 0, 0
    if LAST_PATH.is_file():
        state = torch.load(LAST_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
        optimizer.load_state_dict(state["optimizer"])
        best_miou, best_tag, best_state, best_per_class = state["best_miou"], state["best_tag"], state["best_state"], state["best_per_class"]
        step, start_epoch, history, log_lines = state["step"], state["epoch"] + 1, state["history"], state["log_lines"]
        print(f"Resumed from {LAST_PATH}: {start_epoch} epochs done, best val mIoU {best_miou:.4f} ({best_tag}); earlier epochs:")
        for line in log_lines:
            print(line)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_start = time.time()
        loss_sum, seen = 0.0, 0
        train_mious: list[float] = []
        train_accs: list[float] = []
        for images, masks in sam_train_loader:
            lr = lr_at(step=step, total_steps=total_steps, warmup_steps=warmup_steps, cfg=cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, aux = model(images)
            loss = seg_loss(logits=logits, targets=masks, cfg=cfg)
            if aux is not None and cfg.aux_weight > 0:
                loss = loss + cfg.aux_weight * F.cross_entropy(aux.float(), masks, ignore_index=IGNORE_INDEX)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            ema.update(model=model)
            loss_sum += loss.item() * images.shape[0]
            seen += images.shape[0]
            step += 1
            if step % 4 == 0:  # training mIoU / accuracy on every 4th batch (augmented images, raw weights)
                with torch.no_grad():
                    conf = confusion_per_image(pred=logits.argmax(dim=1), target=masks)
                    train_mious.append(miou_from_confusion(conf=conf).mean().item())
                    total = conf.sum(dim=0)
                    train_accs.append((total.diagonal().sum() / total.sum().clamp(min=1)).item())
        train_loss = loss_sum / max(seen, 1)
        epoch_minutes = (time.time() - epoch_start) / 60

        raw_miou, raw_acc, raw_loss, raw_per_class = evaluate(model=model, loader=sam_val_loader, device=device, cfg=cfg)
        raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
        ema_miou, ema_acc, ema_loss, ema_per_class = evaluate(model=model, loader=sam_val_loader, device=device, cfg=cfg)
        model.load_state_dict(raw_state)
        if ema_miou >= raw_miou:
            epoch_best, tag, per_class = ema_miou, "ema", ema_per_class
        else:
            epoch_best, tag, per_class = raw_miou, "raw", raw_per_class
        record = epoch_best > best_miou
        if record:
            best_miou, best_tag, best_per_class = epoch_best, tag, per_class
            best_state = ema.state_dict() if tag == "ema" else raw_state

        line = " ".join([
            "Epoch:{}/{}..".format(epoch + 1, cfg.epochs),
            "Train Loss: {:.3f}..".format(train_loss),
            "Val Loss: {:.3f}..".format(raw_loss),
            "Train mIoU:{:.3f}..".format(float(np.mean(train_mious)) if train_mious else 0.0),
            "Val mIoU: {:.3f}..".format(raw_miou),
            "Train Acc:{:.3f}..".format(float(np.mean(train_accs)) if train_accs else 0.0),
            "Val Acc:{:.3f}..".format(raw_acc),
            "Time: {:.2f}m".format(epoch_minutes),
            "| EMA Val mIoU: {:.4f} | LR: {:.2e} | min: {:.1f}{}".format(ema_miou, lr, epoch_minutes, " *" if record else ""),
        ])
        print(line)
        log_lines.append(line)
        for key, value in zip(keys, (train_loss, raw_loss, float(np.mean(train_mious)) if train_mious else 0.0, raw_miou,
                                     float(np.mean(train_accs)) if train_accs else 0.0, raw_acc, lr, ema_miou)):
            history[key].append(value)
        if epoch == start_epoch:
            print(f"  ETA: about {(cfg.epochs - epoch - 1) * (time.time() - epoch_start) / 60:.0f} min for the remaining {cfg.epochs - epoch - 1} epochs")
        save_state(
            path=LAST_PATH,
            model=model,
            ema=ema,
            optimizer=optimizer,
            epoch=epoch,
            step=step,
            best_miou=best_miou,
            best_tag=best_tag,
            best_state=best_state,
            best_per_class=best_per_class,
            history=history,
            log_lines=log_lines,
        )

    assert best_state is not None and best_per_class is not None
    print(f"Best val mIoU: {best_miou:.4f} ({best_tag})")
    print(f"  per-class IoU ({best_tag}): {format_per_class(per_class=best_per_class)}")
    model.load_state_dict(best_state)
    tta_miou, tta_acc, _, _ = evaluate(model=model, loader=sam_val_loader, device=device, cfg=cfg, tta_hflip=True)
    print(f"Val mIoU with hflip TTA: {tta_miou:.4f} (acc {tta_acc:.4f}); without: {best_miou:.4f}")
    torch.save({"config": asdict(cfg), "state_dict": best_state, "val_miou": best_miou, "tag": best_tag}, CKPT_PATH)
    print(f"Saved {CKPT_PATH} | elapsed {(time.time() - start) / 60:.1f} min")
    return history
'''

RUN_CELL = r'''
history_sam = train_minisam(cfg=CONFIG)
'''

PLOT_CELL = r'''
plot_loss(history_sam)
plot_score(history_sam)
'''

INFER_CELL = r'''
def set_decoder_grid(model: nn.Module, size: int) -> None:
    """Resize the decoder's learned stride-16 positional embedding to the token grid of another input size."""
    decoder = model.decoder
    if not hasattr(decoder, "image_pe_native"):
        decoder.image_pe_native = decoder.image_pe.detach().clone()
    native = decoder.image_pe_native
    grid = int(round(native.shape[1] ** 0.5))
    target = size // 16
    if target == grid:
        pe = native
    else:
        pe = native.view(1, grid, grid, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(target, target), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, target * target, -1)
    decoder.image_pe = nn.Parameter(pe.contiguous(), requires_grad=False)


@torch.no_grad()
def predict_probs(model: nn.Module, images: torch.Tensor, sizes: tuple[int, ...], tta: bool) -> torch.Tensor:
    """Softmax averaged over the given input sizes (bilinear resize in, logits resized back) and the horizontal flip."""
    h, w = images.shape[-2:]
    probs = None
    for size in sizes:
        x = images if size == h else F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
        set_decoder_grid(model=model, size=size)
        logits = predict_logits(model=model, images=x, tta_hflip=tta)
        if size != h:
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        p = F.softmax(logits, dim=1)
        probs = p if probs is None else probs + p
    set_decoder_grid(model=model, size=CONFIG.img_size)
    return probs / len(sizes)


def suppress_small_classes(probs: torch.Tensor, min_area: int) -> torch.Tensor:
    """Per image, zero the probability of every non-background class predicted on fewer than min_area pixels: a class
    predicted by mistake adds a zero IoU term to that image's mean, so tiny spurious regions cost more than they gain."""
    if min_area <= 0:
        return probs
    probs = probs.clone()
    pred = probs.argmax(dim=1)
    counts = torch.stack([(pred == c).flatten(1).sum(dim=1) for c in range(probs.shape[1])], dim=1)  # (B, C)
    small = (counts > 0) & (counts < min_area)
    small[:, 0] = False
    probs[small.unsqueeze(-1).unsqueeze(-1).expand_as(probs)] = 0.0
    return probs


@torch.no_grad()
def evaluate_chain(model: nn.Module, loader: DataLoader, sizes: tuple[int, ...], min_area: int) -> float:
    """Mean per-image mIoU on validation images given at the competition resolution (ORIG_SIZE)."""
    model.eval()
    mious: list[torch.Tensor] = []
    for images, masks in loader:
        probs = predict_probs(model=model, images=images.to(device, non_blocking=True), sizes=sizes, tta=True)
        probs = suppress_small_classes(probs=probs, min_area=min_area)
        conf = confusion_per_image(pred=probs.argmax(dim=1), target=masks.to(device, non_blocking=True))
        mious.append(miou_from_confusion(conf=conf))
    return torch.cat(mious).mean().item()


model = MiniSAM(cfg=CONFIG).to(device)
model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=False)["state_dict"])
model.eval()
chain_val_set = CIHPSegDataset(
    image_dir=DATA_DIR / "val",
    mask_dir=DATA_DIR / "val_mask",
    ids=list_ids(folder=DATA_DIR / "val")[:SMOKE_SUBSET],
    transform=None,
    hflip_prob=0.0,
    size=ORIG_SIZE,
    resize_mask=False,
)
chain_loader = DataLoader(dataset=chain_val_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
print(f"val mIoU {CONFIG.img_size} hflip: {evaluate_chain(model=model, loader=chain_loader, sizes=(CONFIG.img_size,), min_area=0):.4f}")
print(f"val mIoU {len(INFER_SIZES)} sizes: {evaluate_chain(model=model, loader=chain_loader, sizes=INFER_SIZES, min_area=0):.4f}")
upload_miou = evaluate_chain(model=model, loader=chain_loader, sizes=INFER_SIZES, min_area=MIN_AREA)
print(f"val mIoU {len(INFER_SIZES)} sizes min-area {MIN_AREA}: {upload_miou:.4f}")
print(f"upload setting (sizes {list(INFER_SIZES)}, hflip, min-area {MIN_AREA}): val mIoU {upload_miou:.4f}")
'''

CHECKPOINT_CELL = r'''
checkpoint = CKPT_PATH  # mini-SAM weights of the best epoch (by val mIoU), written by train_minisam above
if not checkpoint.is_file():
    raise FileNotFoundError(f"Run the mini-SAM training first to create {checkpoint}")
model = MiniSAM(cfg=CONFIG)
model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"])
model = model.to(device).eval()
'''

PREDICT_CELL = r'''
def predict_image(model, image, mean=MEAN, std=STD):
    """One 320 x 320 PIL image -> (320, 320) class map with the inference chain evaluated above: softmax averaged
    over the INFER_SIZES inputs and their horizontal flips, then classes predicted on fewer than MIN_AREA pixels dropped."""
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = predict_probs(model=model, images=image, sizes=INFER_SIZES, tta=True)
        probs = suppress_small_classes(probs=probs, min_area=MIN_AREA)
    return probs.argmax(dim=1).cpu().squeeze(0)
'''

COPY_CELL = r'''
if ON_COLAB:  # keep a copy next to the checkpoints so the CSV survives the session
    shutil.copy(output_path, DRIVE_DIR / "runs" / "output.csv")
    print("Copied to", DRIVE_DIR / "runs" / "output.csv")
'''


def intro(cfg_src: str, epoch_min: float, img_size: int, epochs: int) -> str:
    loss = "Lovász-softmax (per image, over the classes present)" if "lovasz_weight" in cfg_src else "Dice"
    hours = epoch_min * epochs / 60
    return f"""
# Mini-SAM: the final model

The U-Net above is the required baseline. The model that produces `output.csv` is a small SAM-style segmenter trained
from scratch on the provided training set only (no pretrained weights, no external data, one model, one training run):

- **Encoder**: stride-4 convolutional stem, then a four-stage hierarchical transformer (windowed attention at strides
  4 and 8, global attention at 16 and 32, depthwise-convolution positional encoding, LayerScale, stochastic depth).
- **Neck + decoder**: an FPN neck and a SAM two-way-transformer mask decoder whose 20 learnable class tokens predict
  the class masks at stride 4 (plus an auxiliary stride-8 head during training).
- **Recipe**: loss 0.5 CE + 0.5 {loss} + 0.4 auxiliary CE; AdamW, linear warmup then cosine decay, EMA weights,
  bf16 autocast, label-aware horizontal flip and affine augmentation; input {img_size} x {img_size}
  (the data are 320 x 320); the epoch with the best validation per-image mIoU (the graded metric) is kept.
- **Resumable**: the full training state is written to `LAST_PATH` after every epoch (on Google Drive when run on
  Colab). If the session is reclaimed, run the notebook again from the top: the training cell resumes and replays the
  earlier epoch lines, so its output is the complete record.
- **Time**: about {epoch_min:.1f} min per epoch on an RTX PRO 6000 (Colab G4), about {hours:.1f} h for {epochs} epochs,
  plus a few minutes for the inference chain and the CSV.
"""


def main() -> int:
    args = sys.argv[1:]
    force = "--force" in args
    sets: dict[str, str] = {}
    infer_sizes = "320, 384, 448, 512"
    epoch_min = 2.8
    positional: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--set":
            k, v = args[i + 1].split("=", 1)
            sets[k] = v
            i += 2
        elif args[i] == "--infer-sizes":
            infer_sizes = ", ".join(args[i + 1].split(","))
            i += 2
        elif args[i] == "--epoch-min":
            epoch_min = float(args[i + 1])
            i += 2
        elif args[i] == "--force":
            i += 1
        else:
            positional.append(args[i])
            i += 1
    script, out = Path(positional[0]), Path(positional[1])
    if out.exists() and not force:
        print(f"refusing to overwrite {out}; pass --force", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "base.ipynb"
        subprocess.run([sys.executable, str(HERE / "build_notebook.py"), str(base), "--force"], check=True, stdout=subprocess.DEVNULL)
        nb = json.loads(base.read_text())
    cells: list[dict] = nb["cells"]

    source = script.read_text()
    head, _, _ = source.partition('\nif __name__ == "__main__":\n')
    sections = dict(split_sections(head))

    # imports of the script (the notebook already has most of them; duplicates are harmless)
    imports = [l for l in sections["Preamble"].splitlines() if l.startswith(("import ", "from ")) and "__future__" not in l and "argparse" not in l]

    experiment = section(sections, "Experiment")
    experiment = re.sub(r'^EXPERIMENT_NAME: str = ".*"$', 'EXPERIMENT_NAME: str = "minisam-final"', experiment, flags=re.M)
    experiment = re.sub(r"^CURRENT_BEST_VAL: .*\n", "", experiment, flags=re.M)
    experiment = re.sub(r"^WRITE_OUTPUTS: .*\n", "", experiment, flags=re.M)
    for k, v in sets.items():
        experiment = set_default(experiment, k, v)
    experiment = experiment.rstrip("\n") + f"""
INFER_SIZES: tuple[int, ...] = ({infer_sizes})  # input sizes averaged at test time (multiples of 64; the decoder's positional embedding is resized)
MIN_AREA: int = 50  # per image, drop any non-background class predicted on fewer pixels than this
SMOKE_SUBSET: int | None = None  # scripts/tools/smoke_final_notebook.py sets this to run on a few images
"""
    img_size = int(re.search(r"^    img_size: int = (\d+)", experiment, re.M).group(1))
    epochs = int(re.search(r"^    epochs: int = (\d+)", experiment, re.M).group(1))

    paths = section(sections, "Paths")
    paths_lines = paths.splitlines()
    first = next(i for i, l in enumerate(paths_lines) if l.startswith("N_CLASSES"))
    paths = "\n".join(paths_lines[first:]) + "\n"  # drop the LAB_DIR-based paths; the notebook's follow below
    data = section(sections, "Data")
    assert "resize_mask" in data, "the source script must be the 448-style script (CIHPSegDataset with size/resize_mask)"

    sam_cells = [
        markdown(intro(cfg_src=experiment, epoch_min=epoch_min, img_size=img_size, epochs=epochs)),
        markdown("## Imports and constants"),
        code("\n".join(imports) + "\n\n" + paths),
        markdown("## Configuration and paths"),
        code(experiment + "\n" + PATHS_CELL),
        markdown("## Data: resized to the training size, label-aware flip, affine augmentation"),
        code(data),
        markdown("## Model"),
        code(section(sections, "Model")),
        markdown("## Loss and metrics (per-image mIoU, the course protocol)"),
        code(section(sections, "Loss / metrics")),
        markdown("## Optimisation: parameter groups, warmup + cosine schedule, EMA"),
        code(section(sections, "Optimisation")),
        markdown("## Training (resumable)"),
        code(TRAIN_CELL),
        code(RUN_CELL),
        code(PLOT_CELL),
        markdown("## Inference chain: multi-size + flip averaging and small-class suppression, measured on validation"),
        code(INFER_CELL),
    ]

    # --- assemble -------------------------------------------------------------
    cells.insert(find(cells, "## Unzip Data") + 1, code(DATA_CELL))
    cells[find(cells, "## Unzip Data")]["source"] = markdown(
        "## Unzip Data\n\nOn Colab the next cell mounts Google Drive and unpacks the five zips from `DRIVE_DIR/data`; "
        "elsewhere the prepared `data/` folder beside this notebook is used.\n\n"
        + "".join(cells[find(cells, "## Unzip Data")]["source"]).split("\n", 2)[2]
    )["source"]
    at = find_start(cells, "plot_loss(history)") + 1  # after the U-Net curves
    cells[at:at] = sam_cells
    predict_md = find(cells, "### Predict Result")
    cells[predict_md]["source"] = markdown(
        "".join(cells[predict_md]["source"]).replace(
            "Load the best validation-loss checkpoint and predict the **course test set**.",
            "Load the mini-SAM weights of the best validation-mIoU epoch and predict the **course test set** with the inference chain measured above.",
        )
    )["source"]
    cells[find(cells, "checkpoint = OUTPUT_DIR / 'Unet-CIHP-best.pt'")] = code(CHECKPOINT_CELL)
    cells[find(cells, "def predict_image(model, image")] = code(PREDICT_CELL)
    cells.insert(find(cells, "print('Saved:', output_path)") + 1, code(COPY_CELL))

    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out}: {len(cells)} cells from {script.name} (img_size {img_size}, epochs {epochs}, infer sizes {infer_sizes})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
