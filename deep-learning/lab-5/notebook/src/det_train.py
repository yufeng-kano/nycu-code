"""Training loop for the anchor-free detector, as used by the final notebook.

The recipe is the one that won the experiments in `scripts/train` (run 015: ConvNeXt-B, PAFPN width 384 /
depth 3, batch 12, copy-paste): 800 letterbox with batch-level multi-scale 0.6-1.25, mosaic + copy-paste +
mixup + random affine + HSV + hflip, mosaic switched off for the last `close_mosaic` epochs, task-aligned assignment with CIoU / DFL losses, AdamW (lower
backbone learning rate, no weight decay on norms and biases), warmup + cosine schedule, bf16
autocast, EMA weights evaluated every `eval_interval` epochs and the best-AP EMA weights kept (or, with
`select = "last"`, the EMA weights of the last epoch). With `train_on_val` the training set is train.txt plus
val.txt minus a held-out slice of `holdout` val images, which then serves as the (small) validation set.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from .config import NUM_CLASSES
from .det_data import AugmentConfig, DetectionDataset, detection_collate
from .det_eval import evaluate_detector
from .det_loss import DetectionLoss
from .detector import Detector


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2026
    # data
    image_size: int = 800
    multi_scale: tuple[float, float] = (0.6, 1.25)
    multi_scale_every: int = 10
    mosaic: float = 1.0
    mixup: float = 0.15
    copy_paste: float = 0.5  # probability of pasting 1..copy_paste_objects object crops onto a mosaic
    copy_paste_objects: int = 4
    affine_scale: tuple[float, float] = (0.5, 1.5)
    affine_translate: float = 0.1
    close_mosaic: int = 10
    batch_size: int = 12
    num_workers: int = 12
    # model
    model_name: str = "convnext_base.fb_in1k"
    width: int = 384
    depth: int = 3
    reg_max: int = 16
    drop_path_rate: float = 0.4
    grad_checkpointing: bool = True
    # assigner / loss
    top_k: int = 10
    alpha: float = 0.5
    beta: float = 6.0
    box_weight: float = 7.5
    cls_weight: float = 0.5
    dfl_weight: float = 1.5
    # optimisation
    epochs: int = 40
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
    # final model
    train_on_val: bool = False  # train on train.txt + val.txt minus `holdout` val images
    holdout: int = 400  # val images kept out of training when train_on_val (seeded split)
    select: str = "best"  # "best": keep the EMA weights with the best validation AP; "last": the last epoch's


# ============================================================================ #
#                                  Utilities                                   #
# ============================================================================ #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


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
#                                    Data                                      #
# ============================================================================ #
def split_lists(cfg: TrainConfig, data_root: Path, split_dir: Path) -> tuple[Path, Path]:
    """Write `trainval.txt` (train + val minus the hold-out) and `holdout.txt` (a seeded slice of val) to split_dir."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    train_lines = (data_root / "train.txt").read_text().splitlines()
    val_lines = (data_root / "val.txt").read_text().splitlines()
    random.Random(cfg.seed).shuffle(val_lines)
    holdout, extra = val_lines[: cfg.holdout], val_lines[cfg.holdout :]
    trainval_path, holdout_path = split_dir / "trainval.txt", split_dir / "holdout.txt"
    trainval_path.write_text("\n".join(train_lines + extra) + "\n")
    holdout_path.write_text("\n".join(holdout) + "\n")
    return trainval_path, holdout_path


def build_datasets(
    cfg: TrainConfig,
    data_root: Path,
    split_dir: Path | None = None,
) -> tuple[DetectionDataset, DetectionDataset, DetectionDataset]:
    """Train / val / test datasets; with cfg.train_on_val the lists come from `split_lists` (written to split_dir)."""
    augment = AugmentConfig(
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        scale=cfg.affine_scale,
        translate=cfg.affine_translate,
        copy_paste=cfg.copy_paste,
        copy_paste_objects=cfg.copy_paste_objects,
    )
    train_list, val_list = data_root / "train.txt", data_root / "val.txt"
    if cfg.train_on_val:
        if split_dir is None:
            raise ValueError("train_on_val needs split_dir for the trainval / holdout lists")
        train_list, val_list = split_lists(cfg, data_root, split_dir)
    train_dataset = DetectionDataset(
        root=data_root,
        list_file=train_list,
        image_size=cfg.image_size,
        train=True,
        augment=augment,
    )
    val_dataset = DetectionDataset(root=data_root, list_file=val_list, image_size=cfg.image_size, train=False)
    test_dataset = DetectionDataset(root=data_root, list_file=data_root / "test.txt", image_size=cfg.image_size, train=False)
    return train_dataset, val_dataset, test_dataset


def make_loader(dataset: DetectionDataset, cfg: TrainConfig, shuffle: bool, limit: int | None = None) -> DataLoader:
    source = Subset(dataset, list(range(limit))) if limit else dataset
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


def subset_coco(dataset: DetectionDataset, n_images: int) -> dict:
    """The COCO dict restricted to the first n images (for smoke tests)."""
    keep = {record["id"] for record in dataset.images[:n_images]}
    return {
        **dataset.coco,
        "images": [record for record in dataset.images if record["id"] in keep],
        "annotations": [ann for ann in dataset.coco["annotations"] if ann["image_id"] in keep],
    }


# ============================================================================ #
#                                 Optimisation                                 #
# ============================================================================ #
def build_optimizer(model: Detector, cfg: TrainConfig) -> torch.optim.Optimizer:
    groups: dict[tuple[bool, bool], list[Tensor]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        no_decay = parameter.ndim <= 1 or name.endswith(".bias")
        groups.setdefault((is_backbone, no_decay), []).append(parameter)
    param_groups = []
    for (is_backbone, no_decay), params in groups.items():
        base_lr = cfg.lr * (cfg.backbone_lr_mult if is_backbone else 1.0)
        param_groups.append(
            {
                "params": params,
                "lr": base_lr,
                "weight_decay": 0.0 if no_decay else cfg.weight_decay,
                "base_lr": base_lr,
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


def random_scale_size(cfg: TrainConfig) -> int:
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
def build_model(cfg: TrainConfig, pretrained: bool = True) -> Detector:
    return Detector(
        model_name=cfg.model_name,
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
        width=cfg.width,
        depth=cfg.depth,
        reg_max=cfg.reg_max,
        image_size=cfg.image_size,
        drop_path_rate=cfg.drop_path_rate,
        grad_checkpointing=cfg.grad_checkpointing,
    )


def train_detector(
    cfg: TrainConfig,
    data_root: Path,
    checkpoint_dir: Path,
    run_name: str,
    device: torch.device,
    log: Callable[[str], None] = print,
    smoke: bool = False,
) -> tuple[Detector, list[dict]]:
    """Train from ImageNet weights (or resume from `<run_name>-last.pt`); returns the best-AP EMA model and the history.

    Checkpoints: `<checkpoint_dir>/<run_name>.pt` (best EMA weights, state_dict) and `<run_name>-last.pt` (resumable).
    """
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False  # multi-scale inputs would re-benchmark every new shape
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"{run_name}.pt"
    last_path = checkpoint_dir / f"{run_name}-last.pt"
    log(f"[{run_name}] config: {json.dumps(asdict(cfg))}")

    train_dataset, val_dataset, _ = build_datasets(cfg, data_root, split_dir=checkpoint_dir)
    log(f"train {len(train_dataset):,} | val {len(val_dataset):,}")
    limit = cfg.batch_size * 2 if smoke else None
    train_loader = make_loader(train_dataset, cfg, shuffle=True, limit=limit)
    val_loader = make_loader(val_dataset, cfg, shuffle=False, limit=limit)
    coco_data = subset_coco(val_dataset, limit) if smoke else val_dataset.coco

    model = build_model(cfg, pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    log(f"params {n_params / 1e6:.2f}M (backbone {n_backbone / 1e6:.2f}M) | {cfg.model_name}")
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
    if not smoke and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        ema.load_state_dict(state["ema"])
        start_epoch, best_ap, best_ap_epoch, history = state["epoch"], state["best_ap"], state["best_ap_epoch"], state["history"]
        log(f"resumed from {last_path.name} at epoch {start_epoch}")

    run_start = time.time()
    current_size = cfg.image_size
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

            with torch.autocast(device.type, dtype=torch.bfloat16):
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
                log(
                    f"  epoch {epoch + 1} step {step}/{steps_per_epoch} | loss {loss.item():.3f}"
                    f" | box {loss_dict['box'].item():.3f} cls {loss_dict['cls'].item():.3f}"
                    f" dfl {loss_dict['dfl'].item():.3f} | fg {loss_dict['num_fg'].item():.0f} | size {current_size}"
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
            record.update(metrics)
            line += (
                f" | AP {metrics['AP']:.4f} | AP50 {metrics['AP50']:.4f} | AP75 {metrics['AP75']:.4f}"
                f" | APs {metrics['AP_small']:.4f} | APm {metrics['AP_medium']:.4f} | APl {metrics['AP_large']:.4f}"
            )
            is_best = metrics["AP"] > best_ap
            if is_best:
                best_ap, best_ap_epoch = metrics["AP"], epoch + 1
            if is_best or cfg.select == "last":  # "*" marks the weights that are saved as the model
                torch.save(ema.module.state_dict(), best_path)
                line += " *"
        log(line)
        history.append(record)

        if not smoke:
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
                last_path,
            )

    kept = "last epoch" if cfg.select == "last" else f"epoch {best_ap_epoch}"
    log(f"done: best val AP {best_ap:.4f} at epoch {best_ap_epoch} | kept {kept} | elapsed {(time.time() - run_start) / 60:.1f} min")
    last_per_class = next((h["per_class_ap"] for h in reversed(history) if "per_class_ap" in h), {})
    if last_per_class:
        log("per-class AP (last eval): " + ", ".join(f"{k} {v:.3f}" for k, v in last_per_class.items() if v is not None))

    best_model = build_model(cfg, pretrained=False).to(device)
    best_model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    return best_model.eval(), history
