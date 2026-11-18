"""DL Lab 5 experiment 006-size-800: 005 with an 800 letterbox instead of 640.

Why: evaluating the 005 weights (trained at 640 with multi-scale 0.75-1.25) at larger inputs raises
val AP from 0.401 to 0.409 at 704/768 and keeps lifting small-object AP up to 896, so the detector is
starved of resolution on small objects (fork / knife / spoon / remote / mouse). Here the base size is
800 and the batch multi-scale range becomes 0.6-1.0, so the training sizes stay in [480, 800] (same
compute and memory as 005) while the mosaic canvas, letterbox and evaluation move to 800.
Everything else is identical to 005.

Model (notebook/src/detector.py): timm convnext_tiny.fb_in1k backbone (ImageNet-1k only) -> PAFPN neck
(width 256, CSP depth 2) -> decoupled head with DFL box regression (reg_max 16) at strides 8/16/32.
Assignment and loss (notebook/src/det_loss.py): task-aligned assigner (top-k 10, alpha 0.5, beta 6),
BCE classification against the alignment scores, CIoU box loss, distribution focal loss (7.5 / 0.5 / 1.5).
Data (notebook/src/det_data.py): 640 letterbox, mosaic + random affine (scale 0.5-1.5, translate 0.1)
+ HSV + hflip; mosaic switched off for the last `close_mosaic` epochs; batch-level multi-scale
training in [0.75, 1.25] x 640 (multiples of 32), changed every 10 iterations.
Optimisation: AdamW (lr 4e-4 neck/head, 1e-4 backbone, wd 0.05, none on norms/biases), 3 warmup
epochs then cosine to 1e-6, grad clip 10, bf16 autocast, EMA 0.9998; val AP every `eval_interval`
epochs on the EMA weights at conf 0.001 / NMS 0.65 / 100 boxes; best-AP checkpoint kept.
Resumable through data/checkpoints/<name>-last.pt.

Run from anywhere:  python scripts/train/006-size-800.py            (full run or resume)
                    python scripts/train/006-size-800.py --smoke    (a few steps + tiny eval, no outputs)
"""

from __future__ import annotations

import argparse
import copy
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
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

# ============================================================================ #
#                                Experiment                                    #
# ============================================================================ #
EXPERIMENT_NAME: str = "006-size-800"
WRITE_OUTPUTS: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    # data
    image_size: int = 800
    multi_scale: tuple[float, float] = (0.6, 1.0)
    multi_scale_every: int = 10
    mosaic: float = 1.0
    mixup: float = 0.15
    affine_scale: tuple[float, float] = (0.5, 1.5)
    affine_translate: float = 0.1
    close_mosaic: int = 10
    batch_size: int = 16
    num_workers: int = 12
    # model
    model_name: str = "convnext_tiny.fb_in1k"
    width: int = 256
    depth: int = 2
    reg_max: int = 16
    drop_path_rate: float = 0.3
    # assigner / loss
    top_k: int = 10
    alpha: float = 0.5
    beta: float = 6.0
    box_weight: float = 7.5
    cls_weight: float = 0.5
    dfl_weight: float = 1.5
    # optimisation
    epochs: int = 60
    warmup_epochs: int = 3
    lr: float = 4e-4
    backbone_lr_mult: float = 0.25
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    grad_clip: float = 10.0
    ema_decay: float = 0.9998
    ema_tau: int = 2000
    # evaluation
    eval_interval: int = 5
    conf_thres: float = 0.001
    nms_thres: float = 0.65
    tta_hflip: bool = True


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
BEST_PATH: Path = CKPT_DIR / f"{EXPERIMENT_NAME}.pt"
LAST_PATH: Path = CKPT_DIR / f"{EXPERIMENT_NAME}-last.pt"

sys.path.insert(0, str(NOTEBOOK_DIR))
from src.config import NUM_CLASSES  # noqa: E402
from src.det_data import AugmentConfig, DetectionDataset, detection_collate  # noqa: E402
from src.det_eval import evaluate_detector, write_submission  # noqa: E402
from src.det_loss import DetectionLoss  # noqa: E402
from src.detector import Detector  # noqa: E402


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


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


# ============================================================================ #
#                                    EMA                                       #
# ============================================================================ #
class ModelEMA:
    """Exponential moving average of the weights with a warm-up ramp on the decay."""

    def __init__(self, model: nn.Module, decay: float, tau: int) -> None:
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1 - math.exp(-self.updates / self.tau))
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                ema_state[key].mul_(decay).add_(value.detach(), alpha=1 - decay)
            else:
                ema_state[key].copy_(value)

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = state["updates"]


# ============================================================================ #
#                                   Data                                       #
# ============================================================================ #
def build_datasets(cfg: ExperimentConfig) -> tuple[DetectionDataset, DetectionDataset, DetectionDataset]:
    augment = AugmentConfig(
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        scale=cfg.affine_scale,
        translate=cfg.affine_translate,
    )
    train_dataset = DetectionDataset(
        root=DATA_ROOT,
        list_file=DATA_ROOT / "train.txt",
        image_size=cfg.image_size,
        train=True,
        augment=augment,
    )
    val_dataset = DetectionDataset(
        root=DATA_ROOT,
        list_file=DATA_ROOT / "val.txt",
        image_size=cfg.image_size,
        train=False,
    )
    test_dataset = DetectionDataset(
        root=DATA_ROOT,
        list_file=DATA_ROOT / "test.txt",
        image_size=cfg.image_size,
        train=False,
    )
    return train_dataset, val_dataset, test_dataset


def make_loader(dataset, cfg: ExperimentConfig, shuffle: bool, smoke_count: int | None) -> DataLoader:
    source = Subset(dataset, list(range(smoke_count))) if smoke_count else dataset
    return DataLoader(
        source,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=worker_init,
    )


def smoke_coco(val_dataset: DetectionDataset, n_images: int) -> dict:
    keep = {record["id"] for record in val_dataset.images[:n_images]}
    return {
        **val_dataset.coco,
        "images": [record for record in val_dataset.images if record["id"] in keep],
        "annotations": [ann for ann in val_dataset.coco["annotations"] if ann["image_id"] in keep],
    }


# ============================================================================ #
#                                 Optimisation                                 #
# ============================================================================ #
def build_optimizer(model: Detector, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    groups: dict[tuple[bool, bool], list[Tensor]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        no_decay = parameter.ndim <= 1 or name.endswith(".bias")
        groups.setdefault((is_backbone, no_decay), []).append(parameter)
    param_groups = []
    for (is_backbone, no_decay), params in groups.items():
        param_groups.append(
            {
                "params": params,
                "lr": cfg.lr * (cfg.backbone_lr_mult if is_backbone else 1.0),
                "weight_decay": 0.0 if no_decay else cfg.weight_decay,
                "base_lr": cfg.lr * (cfg.backbone_lr_mult if is_backbone else 1.0),
            }
        )
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def lr_factor(step: int, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, factor: float) -> float:
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * factor
    return optimizer.param_groups[0]["lr"]


def random_scale_size(cfg: ExperimentConfig) -> int:
    """A random multiple of 32 within the multi-scale range."""
    low = int(cfg.image_size * cfg.multi_scale[0] / 32) * 32
    high = int(cfg.image_size * cfg.multi_scale[1] / 32) * 32
    return random.randrange(low, high + 1, 32)


def resize_batch(images: Tensor, targets: Tensor, size: int) -> tuple[Tensor, Tensor]:
    """Resize the whole batch to size x size; boxes follow."""
    if size == images.shape[-1]:
        return images, targets
    ratio = size / images.shape[-1]
    images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    targets = targets.clone()
    targets[:, 2:6] *= ratio
    return images, targets


# ============================================================================ #
#                                  Training                                    #
# ============================================================================ #
def main(cfg: ExperimentConfig, smoke: bool) -> None:
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False  # multi-scale inputs would re-benchmark every new shape (minutes per shape for depthwise convs)
    device = torch.device("cuda:0")
    print(f"[{EXPERIMENT_NAME}] config: {json.dumps(asdict(cfg))}")

    train_dataset, val_dataset, test_dataset = build_datasets(cfg)
    smoke_n = cfg.batch_size * 2 if smoke else None
    train_loader = make_loader(train_dataset, cfg, shuffle=True, smoke_count=smoke_n)
    val_loader = make_loader(val_dataset, cfg, shuffle=False, smoke_count=smoke_n)
    print(f"train {len(train_loader.dataset):,} | val {len(val_loader.dataset):,} | test {len(test_dataset):,}")

    model = Detector(
        model_name=cfg.model_name,
        num_classes=NUM_CLASSES,
        pretrained=True,
        width=cfg.width,
        depth=cfg.depth,
        reg_max=cfg.reg_max,
        image_size=cfg.image_size,
        drop_path_rate=cfg.drop_path_rate,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    print(f"params {n_params / 1e6:.2f}M (backbone {n_backbone / 1e6:.2f}M) | {cfg.model_name}")
    criterion = DetectionLoss(
        num_classes=NUM_CLASSES,
        reg_max=cfg.reg_max,
        box_weight=cfg.box_weight,
        cls_weight=cfg.cls_weight,
        dfl_weight=cfg.dfl_weight,
        top_k=cfg.top_k,
        alpha=cfg.alpha,
        beta=cfg.beta,
    )
    optimizer = build_optimizer(model, cfg)
    ema = ModelEMA(model, decay=cfg.ema_decay, tau=cfg.ema_tau)

    epochs = 2 if smoke else cfg.epochs
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    min_ratio = cfg.min_lr / cfg.lr

    start_epoch = 0
    best_ap = -1.0
    best_ap_epoch = 0
    history: list[dict] = []
    if not smoke and LAST_PATH.exists():
        state = torch.load(LAST_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        start_epoch = state["epoch"]
        best_ap = state["best_ap"]
        best_ap_epoch = state["best_ap_epoch"]
        history = state["history"]
        print(f"resumed from {LAST_PATH.name} at epoch {start_epoch}")

    coco_data = smoke_coco(val_dataset, smoke_n) if smoke else val_dataset.coco
    run_start = time.time()
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        train_dataset.mosaic_enabled = epoch < epochs - cfg.close_mosaic
        model.train()
        sums = {"total": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0, "num_fg": 0.0}
        seen = 0
        for step, (images, targets, _) in enumerate(train_loader):
            global_step = epoch * steps_per_epoch + step
            lr = set_lr(optimizer, lr_factor(global_step, warmup_steps, total_steps, min_ratio))
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if step % cfg.multi_scale_every == 0:
                current_size = random_scale_size(cfg)
            scaled_images, scaled_targets = resize_batch(images, targets, current_size)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(scaled_images)
            loss_dict = criterion(outputs, scaled_targets)
            loss = loss_dict["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss is not finite at epoch {epoch + 1} step {step}: {loss_dict}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(model)

            for key in sums:
                sums[key] += loss_dict[key].item() * (1 if key == "num_fg" else len(images))
            seen += len(images)
            if smoke and step >= 3:
                break
            if step % 100 == 0 and not smoke:
                print(
                    f"  epoch {epoch + 1} step {step}/{steps_per_epoch} | loss {loss.item():.3f}"
                    f" | box {loss_dict['box'].item():.3f} cls {loss_dict['cls'].item():.3f}"
                    f" dfl {loss_dict['dfl'].item():.3f} | fg {loss_dict['num_fg'].item():.0f} | size {current_size}",
                    flush=True,
                )

        means = {key: value / max(seen, 1) for key, value in sums.items()}
        line = (
            f"epoch {epoch + 1}/{epochs} | loss {means['total']:.4f} (box {means['box']:.3f} cls {means['cls']:.3f}"
            f" dfl {means['dfl']:.3f}) | fg/img {means['num_fg']:.1f} | lr {lr:.2e}"
            f" | mosaic {'on' if train_dataset.mosaic_enabled else 'off'} | {(time.time() - epoch_start) / 60:.1f} min"
        )
        record = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in means.items()}, "lr": lr}

        if (epoch + 1) % cfg.eval_interval == 0 or epoch + 1 == epochs or smoke:
            metrics = evaluate_detector(
                ema.module,
                val_loader,
                annotation_data=coco_data,
                category_ids=val_dataset.category_ids,
                conf_thres=cfg.conf_thres,
                nms_thres=cfg.nms_thres,
            )
            record.update({k: v for k, v in metrics.items()})
            line += (
                f" | AP {metrics['AP']:.4f} | AP50 {metrics['AP50']:.4f} | AP75 {metrics['AP75']:.4f}"
                f" | APs {metrics['AP_small']:.4f} | APm {metrics['AP_medium']:.4f} | APl {metrics['AP_large']:.4f}"
            )
            if metrics["AP"] > best_ap:
                best_ap, best_ap_epoch = metrics["AP"], epoch + 1
                if WRITE_OUTPUTS and not smoke:
                    torch.save(ema.module.state_dict(), BEST_PATH)
                line += " *"
        print(line, flush=True)
        history.append(record)

        if WRITE_OUTPUTS and not smoke:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema": ema.state_dict(),
                    "epoch": epoch + 1,
                    "best_ap": best_ap,
                    "best_ap_epoch": best_ap_epoch,
                    "history": history,
                    "config": asdict(cfg),
                },
                LAST_PATH,
            )

    print(f"done: best val AP {best_ap:.4f} at epoch {best_ap_epoch} | elapsed {(time.time() - run_start) / 60:.1f} min")
    last_per_class = next((h["per_class_ap"] for h in reversed(history) if "per_class_ap" in h), {})
    if last_per_class:
        print("per-class AP (last eval): " + ", ".join(f"{k} {v:.3f}" for k, v in last_per_class.items() if v is not None))

    # Final: best checkpoint with hflip TTA on val, then the Kaggle CSV.
    if not smoke:
        model.load_state_dict(torch.load(BEST_PATH, map_location="cpu", weights_only=True))
    final_model = model if not smoke else ema.module
    if cfg.tta_hflip:
        tta_metrics = evaluate_detector(
            final_model,
            val_loader,
            annotation_data=coco_data,
            category_ids=val_dataset.category_ids,
            conf_thres=cfg.conf_thres,
            nms_thres=cfg.nms_thres,
            hflip_tta=True,
        )
        print(f"val AP with hflip TTA: {tta_metrics['AP']:.4f} | AP50 {tta_metrics['AP50']:.4f}")
    if smoke or not WRITE_OUTPUTS:
        return
    METRICS_PATH.write_text(json.dumps({"config": asdict(cfg), "history": history}, indent=1))
    test_loader = make_loader(test_dataset, cfg, shuffle=False, smoke_count=None)
    write_submission(
        final_model,
        test_loader,
        CSV_PATH,
        conf_thres=cfg.conf_thres,
        nms_thres=cfg.nms_thres,
        hflip_tta=cfg.tta_hflip,
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
