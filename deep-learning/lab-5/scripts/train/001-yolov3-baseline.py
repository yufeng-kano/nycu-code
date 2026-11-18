"""DL Lab 5 experiment 001-yolov3-baseline: the TA template as shipped, TODOs filled, run as a script.

Reproduces the notebook's training recipe (three-scale YOLOv3 on timm darknet53.c2ns_in1k, 416 input,
hflip only, batch 16, AdamW 1e-4 / wd 5e-4, cosine to 1e-6 over 50 epochs, AMP, GIoU + focal + CE loss)
so that every later run has a measured reference. Differences from the notebook are bookkeeping only:
COCO AP is measured every `eval_interval` epochs and at the end, the checkpoint with the best val AP is
kept next to the template's best-val-loss one, the run is resumable through `-last.pt`, and the Kaggle
CSV is written from the best-AP weights.

Run from anywhere:  python scripts/train/001-yolov3-baseline.py            (full run or resume)
                    python scripts/train/001-yolov3-baseline.py --smoke    (a few steps + tiny eval, no outputs)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

# ============================================================================ #
#                                Experiment                                    #
# ============================================================================ #
EXPERIMENT_NAME: str = "001-yolov3-baseline"
WRITE_OUTPUTS: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    image_size: int = 416
    batch_size: int = 16
    num_workers: int = 8
    epochs: int = 50
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 5e-4
    eval_interval: int = 5
    eval_conf_thres: float = 0.01
    nms_thres: float = 0.4
    lambda_coord: float = 5.0
    lambda_obj: float = 1.0
    lambda_noobj: float = 0.5
    lambda_class: float = 1.0
    box_loss_type: str = "giou"
    model_name: str = "darknet53.c2ns_in1k"


CONFIG = ExperimentConfig()

# ============================================================================ #
#                                  Paths                                       #
# ============================================================================ #
LAB_DIR: Path = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR: Path = LAB_DIR / "notebook"
DATA_ROOT: Path = LAB_DIR / "data" / "dataset"
LOG_PATH: Path = LAB_DIR / "logs" / f"{EXPERIMENT_NAME}.log"
CSV_PATH: Path = LAB_DIR / "result" / f"{EXPERIMENT_NAME}.csv"
METRICS_PATH: Path = LAB_DIR / "result" / f"{EXPERIMENT_NAME}.json"
CKPT_DIR: Path = LAB_DIR / "data" / "checkpoints"
BEST_AP_PATH: Path = CKPT_DIR / f"{EXPERIMENT_NAME}.pt"
BEST_LOSS_PATH: Path = CKPT_DIR / f"{EXPERIMENT_NAME}-best-loss.pt"
LAST_PATH: Path = CKPT_DIR / f"{EXPERIMENT_NAME}-last.pt"

sys.path.insert(0, str(NOTEBOOK_DIR))
from src.config import ANCHORS  # noqa: E402
from src.dataset import CocoDetectorDataset, collate_fn  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.predict_test import predict_test  # noqa: E402
from src.yolo import getODmodel  # noqa: E402
from src.yolo_loss import YOLOv3Loss  # noqa: E402


class Tee:
    """Mirror writes to the console and a log file."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================ #
#                                   Data                                       #
# ============================================================================ #
def build_loaders(cfg: ExperimentConfig, smoke: bool) -> tuple[DataLoader, DataLoader, CocoDetectorDataset]:
    train_dataset = CocoDetectorDataset(
        root=DATA_ROOT,
        list_file=DATA_ROOT / "train.txt",
        train=True,
        encode_target=True,
        image_size=cfg.image_size,
    )
    val_dataset = CocoDetectorDataset(
        root=DATA_ROOT,
        list_file=DATA_ROOT / "val.txt",
        train=False,
        encode_target=True,
        image_size=cfg.image_size,
    )
    train_source = Subset(train_dataset, list(range(cfg.batch_size * 4))) if smoke else train_dataset
    val_source = Subset(val_dataset, list(range(cfg.batch_size * 2))) if smoke else val_dataset
    train_loader = DataLoader(
        train_source,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_source,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader, val_dataset


def smoke_coco(val_dataset: CocoDetectorDataset, n_images: int) -> dict:
    """COCO dict restricted to the first n images, for the smoke evaluation."""
    keep = {record["id"] for record in val_dataset.images[:n_images]}
    return {
        **val_dataset.coco,
        "images": [record for record in val_dataset.images if record["id"] in keep],
        "annotations": [ann for ann in val_dataset.coco["annotations"] if ann["image_id"] in keep],
    }


# ============================================================================ #
#                                  Training                                    #
# ============================================================================ #
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: YOLOv3Loss,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler,
    device: torch.device,
    smoke: bool,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    seen = 0
    with torch.set_grad_enabled(training):
        for step, (images, targets, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = [target.to(device, non_blocking=True) for target in targets]
            with autocast(device.type, enabled=True):
                predictions = model(images)
            loss_dict = criterion(predictions, targets)
            loss = loss_dict["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss is not finite.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += loss.item() * len(images)
            seen += len(images)
            if smoke and step >= 3:
                break
    return total / max(seen, 1)


def main(cfg: ExperimentConfig, smoke: bool) -> None:
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    print(f"[{EXPERIMENT_NAME}] config: {json.dumps(asdict(cfg))}")

    train_loader, val_loader, val_dataset = build_loaders(cfg, smoke)
    print(f"train {len(train_loader.dataset):,} images | val {len(val_loader.dataset):,} images")

    model = getODmodel(pretrained=True, model_name=cfg.model_name).to(device)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | backbone {cfg.model_name}")
    criterion = YOLOv3Loss(
        lambda_coord=cfg.lambda_coord,
        lambda_obj=cfg.lambda_obj,
        lambda_noobj=cfg.lambda_noobj,
        lambda_class=cfg.lambda_class,
        anchors=ANCHORS,
        box_loss_type=cfg.box_loss_type,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.min_lr,
    )
    scaler = GradScaler("cuda", enabled=True)

    start_epoch = 0
    best_val_loss = math.inf
    best_ap = -1.0
    best_ap_epoch = 0
    history: list[dict] = []
    if not smoke and LAST_PATH.exists():
        state = torch.load(LAST_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        best_val_loss = state["best_val_loss"]
        best_ap = state["best_ap"]
        best_ap_epoch = state["best_ap_epoch"]
        history = state["history"]
        print(f"resumed from {LAST_PATH.name} at epoch {start_epoch}")

    epochs = 2 if smoke else cfg.epochs
    coco_data = smoke_coco(val_dataset, len(val_loader.dataset)) if smoke else val_dataset.coco
    run_start = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = run_epoch(model, train_loader, criterion, optimizer, scaler, device, smoke)
        scheduler.step()
        val_loss = run_epoch(model, val_loader, criterion, None, scaler, device, smoke)

        line = (
            f"epoch {epoch + 1}/{epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
            f"| lr {lr:.2e} | {(time.time() - epoch_start) / 60:.1f} min"
        )
        record = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}

        if (epoch + 1) % cfg.eval_interval == 0 or epoch + 1 == epochs or smoke:
            metrics = evaluate(
                model,
                val_loader,
                annotation_data=coco_data,
                category_ids=val_dataset.category_ids,
                conf_thres=cfg.eval_conf_thres,
                use_amp=True,
            )
            record.update({k: v for k, v in metrics.items() if k != "per_class_ap"})
            record["per_class_ap"] = metrics["per_class_ap"]
            line += (
                f" | AP {metrics['AP']:.4f} | AP50 {metrics['AP50']:.4f} | AP75 {metrics['AP75']:.4f}"
                f" | APs {metrics['AP_small']:.4f} | APm {metrics['AP_medium']:.4f} | APl {metrics['AP_large']:.4f}"
            )
            if metrics["AP"] > best_ap:
                best_ap, best_ap_epoch = metrics["AP"], epoch + 1
                if WRITE_OUTPUTS and not smoke:
                    torch.save(model.state_dict(), BEST_AP_PATH)
                line += " *"
        print(line, flush=True)
        history.append(record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if WRITE_OUTPUTS and not smoke:
                torch.save(model.state_dict(), BEST_LOSS_PATH)
        if WRITE_OUTPUTS and not smoke:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "best_ap": best_ap,
                    "best_ap_epoch": best_ap_epoch,
                    "history": history,
                    "config": asdict(cfg),
                },
                LAST_PATH,
            )

    print(
        f"done: best val AP {best_ap:.4f} at epoch {best_ap_epoch} | best val loss {best_val_loss:.4f}"
        f" | elapsed {(time.time() - run_start) / 60:.1f} min"
    )
    last_per_class = next((h["per_class_ap"] for h in reversed(history) if "per_class_ap" in h), {})
    if last_per_class:
        print("per-class AP (last eval): " + ", ".join(f"{k} {v:.3f}" for k, v in last_per_class.items() if v is not None))

    if smoke or not WRITE_OUTPUTS:
        return
    METRICS_PATH.write_text(json.dumps({"config": asdict(cfg), "history": history}, indent=1))
    predict_test(
        weights_path=BEST_AP_PATH,
        data_root=DATA_ROOT,
        output_path=CSV_PATH,
        batch_size=cfg.batch_size,
        conf_thres=cfg.eval_conf_thres,
        nms_thres=cfg.nms_thres,
        device=device,
        model_name=cfg.model_name,
        image_size=cfg.image_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="a few steps + tiny evaluation, no outputs")
    args = parser.parse_args()
    if args.smoke or not WRITE_OUTPUTS:
        main(CONFIG, smoke=args.smoke)
    else:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as log_file:
            sys.stdout = Tee(sys.__stdout__, log_file)
            sys.stderr = Tee(sys.__stderr__, log_file)
            main(CONFIG, smoke=False)
