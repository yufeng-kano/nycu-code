"""DL Lab 3 experiment 037-erasing-0.4: RandomErasing p 0.5 -> 0.4 (base: 016-erasing-0.5).

Self-contained pipeline:
data -> transforms -> model -> supervised -> self-training -> predict.

Run from anywhere: python scripts/037-erasing-0.4.py
"""

import csv
import math
import random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.transforms import v2 as transforms

# ============================================================================ #
#                                Experiment                                    #
# ============================================================================ #
EXPERIMENT_NAME: str = "037-erasing-0.4"
CURRENT_BEST_VAL: float = 0.9325  # val acc of the current best result
WRITE_OUTPUTS: bool = True

ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1),
    "gelu": nn.GELU,
}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    # data / augmentation
    img_size: int = 128
    crop_scale: tuple[float, float] = (0.6, 1.0)
    use_trivial_augment: bool = True
    erasing_p: float = 0.4
    batch_size: int = 64
    num_workers: int = 8
    # model
    activation: str = "gelu"
    stem_channels: int = 32
    widths: tuple[int, ...] = (64, 128, 256, 512)
    depths: tuple[int, ...] = (2, 2, 2, 2)
    se_reduction: int = 16
    drop_path_rate: float = 0.2
    head_dropout: float = 0.2
    use_laurel: bool = False
    # supervised
    sup_epochs: int = 200
    sup_warmup_epochs: int = 5
    sup_lr: float = 1e-3
    sup_min_lr: float = 1e-5
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    # self-training
    ssl_epochs: int = 200
    ssl_warmup_epochs: int = 0
    ssl_lr: float = 5e-4
    ssl_min_lr: float = 1e-5
    ssl_weight_decay: float = 0.05
    ssl_label_smoothing: float = 0.1
    pseudo_interval: int = 5  # N
    threshold: float = 0.9
    notes: list[str] = field(default_factory=list)


CONFIG = ExperimentConfig()

# ============================================================================ #
#                                  Paths                                       #
# ============================================================================ #
LAB_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = LAB_DIR / "Lab3_data_intel_2026"
LOG_PATH: Path = LAB_DIR / "logs" / f"{EXPERIMENT_NAME}.log"
CSV_PATH: Path = LAB_DIR / "result" / f"{EXPERIMENT_NAME}.csv"
CKPT_PATH: Path = LAB_DIR / "checkpoints" / f"{EXPERIMENT_NAME}.pt"

CLASS_NAMES: list[str] = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}


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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================================ #
#                                   Data                                       #
# ============================================================================ #
class IntelData(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "train",
        mode: str = "train",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        use_unlabel: bool = False,
    ) -> None:
        self.root = root
        self.split = split
        self.mode = mode
        self.transform = transform
        self.use_unlabel = use_unlabel

        if split == "train" and use_unlabel:
            csv_file = root / "unlabeled_train.csv"
        elif split == "train":
            csv_file = root / "train.csv"
        elif split == "val":
            csv_file = root / "val.csv"
        else:
            csv_file = root / "test.csv"

        df = pd.read_csv(csv_file)
        self.paths: list[Path] = [root / name for name in df["file_name"]]
        self.labels: list[int] = []
        if "groundtruth" in df.columns:
            self.labels = [CLASS_TO_IDX[g] for g in df["groundtruth"]]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Any:
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.mode == "test" or (self.split == "train" and self.use_unlabel):
            return img
        return img, torch.tensor(self.labels[index], dtype=torch.long)


class PseudoLabelData(Dataset):
    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(self.labels[index], dtype=torch.long)


def compute_mean_std(dataset: Dataset, batch_size: int = 256) -> tuple[list[float], list[float]]:
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    channel_sum: torch.Tensor = torch.zeros(size=(3,), dtype=torch.float64)
    channel_sq_sum: torch.Tensor = torch.zeros(size=(3,), dtype=torch.float64)
    pixel_count: int = 0
    for images, _ in loader:
        images = images.to(dtype=torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += images.pow(2).sum(dim=(0, 2, 3))
        pixel_count += images.size(0) * images.size(2) * images.size(3)
    mean: torch.Tensor = channel_sum / pixel_count
    std: torch.Tensor = (channel_sq_sum / pixel_count - mean.pow(2)).sqrt()
    return mean.tolist(), std.tolist()


def build_transforms(
    cfg: ExperimentConfig,
    mean: list[float],
    std: list[float],
) -> tuple[transforms.Compose, transforms.Compose]:
    train_ops: list[Any] = [
        transforms.RandomResizedCrop(
            size=cfg.img_size,
            scale=cfg.crop_scale,
            ratio=(3 / 4, 4 / 3),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if cfg.use_trivial_augment:
        train_ops.append(transforms.TrivialAugmentWide(num_magnitude_bins=31))
    train_ops += [
        transforms.ToImage(),
        transforms.ToDtype(dtype=torch.float32, scale=True),
        transforms.Normalize(mean=mean, std=std),
    ]
    if cfg.erasing_p > 0:
        train_ops.append(
            transforms.RandomErasing(
                p=cfg.erasing_p,
                scale=(0.02, 0.2),
                value=0,
            ),
        )
    test_ops: list[Any] = [
        transforms.Resize(size=(cfg.img_size, cfg.img_size)),
        transforms.ToImage(),
        transforms.ToDtype(dtype=torch.float32, scale=True),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(train_ops), transforms.Compose(test_ops)


# ============================================================================ #
#                                   Model                                      #
# ============================================================================ #
class ConvBN(nn.Module):
    """Conv2d (no bias) followed by BatchNorm2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch per sample."""

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob: float = 1.0 - self.drop_prob
        mask = torch.empty(size=(x.size(0), 1, 1, 1), device=x.device)
        mask = mask.bernoulli_(p=keep_prob)
        return x * mask / keep_prob


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(
        self,
        channels: int,
        reduction: int,
        activation: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        hidden: int = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=channels, out_channels=hidden, kernel_size=1),
            activation(),
            nn.Conv2d(in_channels=hidden, out_channels=channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class ResidualBlock(nn.Module):
    """Residual block: y = act(a * DropPath(F(x)) + b * shortcut(x))."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        activation: Callable[[], nn.Module],
        se_reduction: int,
        drop_prob: float,
        use_laurel: bool,
    ) -> None:
        super().__init__()
        last_conv = ConvBN(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
        )
        nn.init.zeros_(last_conv.bn.weight)
        self.branch = nn.Sequential(
            ConvBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
            ),
            activation(),
            last_conv,
            SEBlock(
                channels=out_channels,
                reduction=se_reduction,
                activation=activation,
            ),
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride) if stride > 1 else nn.Identity(),
                ConvBN(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                ),
            )

        if use_laurel:
            self.res_weights = nn.Parameter(torch.ones(2))
        else:
            self.register_buffer("res_weights", torch.ones(2))

        self.drop_path = DropPath(drop_prob=drop_prob)
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.drop_path(self.branch(x))
        shortcut = self.shortcut(x)
        return self.act(self.res_weights[0] * branch + self.res_weights[1] * shortcut)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.zeros_(module.bias)


class YourCNNModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig, num_classes: int = 6) -> None:
        super().__init__()
        activation = ACTIVATIONS[cfg.activation]
        self.stem = nn.Sequential(
            ConvBN(
                in_channels=3,
                out_channels=cfg.stem_channels,
                kernel_size=3,
            ),
            activation(),
            ConvBN(
                in_channels=cfg.stem_channels,
                out_channels=cfg.stem_channels,
                kernel_size=3,
            ),
            activation(),
        )

        num_blocks: int = sum(cfg.depths)
        drop_probs: list[float] = torch.linspace(
            start=0.0,
            end=cfg.drop_path_rate,
            steps=num_blocks,
        ).tolist()
        blocks: list[nn.Module] = []
        in_channels: int = cfg.stem_channels
        for width, depth in zip(cfg.widths, cfg.depths):
            for i in range(depth):
                blocks.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        out_channels=width,
                        stride=2 if i == 0 else 1,
                        activation=activation,
                        se_reduction=cfg.se_reduction,
                        drop_prob=drop_probs[len(blocks)],
                        use_laurel=cfg.use_laurel,
                    )
                )
                in_channels = width
        self.stages = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Dropout(p=cfg.head_dropout),
            nn.Linear(in_features=in_channels, out_features=num_classes),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))


# ============================================================================ #
#                              Train / evaluate                                #
# ============================================================================ #
def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Apply weight decay to conv/linear weights only, not to BN, bias or scalars."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim > 1:
            decay.append(param)
        else:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def warmup_cosine_lr(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """LR for a 0-based epoch: linear warmup from 0.1 * base_lr, then cosine."""
    if epoch < warmup_epochs:
        return base_lr * (0.1 + 0.9 * epoch / warmup_epochs)
    progress: float = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss: float = 0.0
    total_count: int = 0
    acc_count: int = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs: torch.Tensor = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predicted: torch.Tensor = outputs.detach().argmax(dim=1)
        total_count += labels.size(0)
        acc_count += int((predicted == labels).sum().item())
        total_loss += loss.item() * labels.size(0)
    return acc_count / total_count, total_loss / total_count


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss: float = 0.0
    total_count: int = 0
    acc_count: int = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs: torch.Tensor = model(images)
        loss = criterion(outputs, labels)
        predicted: torch.Tensor = outputs.argmax(dim=1)
        total_count += labels.size(0)
        acc_count += int((predicted == labels).sum().item())
        total_loss += loss.item() * labels.size(0)
    return acc_count / total_count, total_loss / total_count


@torch.no_grad()
def get_pseudo_labels(
    model: nn.Module,
    unlabeled_set: IntelData,
    unlabeled_indices: list[int],
    threshold: float,
    transform: Callable[[Image.Image], torch.Tensor],
    loader_kwargs: dict[str, Any],
    device: torch.device,
) -> tuple[PseudoLabelData, int]:
    """Label confident samples and remove them from the pool (in place)."""
    model.eval()
    paths_keep: list[Path] = []
    labels_keep: list[int] = []
    remove_indices: list[int] = []

    loader = DataLoader(
        dataset=Subset(unlabeled_set, unlabeled_indices),
        shuffle=False,
        **loader_kwargs,
    )
    offset: int = 0
    for images in loader:
        images = images.to(device, non_blocking=True)
        probs: torch.Tensor = F.softmax(model(images), dim=1)
        confidences, predictions = probs.max(dim=1)
        is_confident: torch.Tensor = confidences >= threshold
        accepted: list[int] = is_confident.nonzero(as_tuple=True)[0].tolist()
        predictions = predictions.cpu()
        for j in accepted:
            position: int = offset + j
            paths_keep.append(unlabeled_set.paths[unlabeled_indices[position]])
            labels_keep.append(int(predictions[j]))
            remove_indices.append(position)
        offset += len(images)

    for index in reversed(remove_indices):
        del unlabeled_indices[index]

    pseudo_dataset = PseudoLabelData(
        paths=paths_keep,
        labels=labels_keep,
        transform=transform,
    )
    return pseudo_dataset, len(pseudo_dataset)


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ============================================================================ #
#                                   Main                                       #
# ============================================================================ #
def main(cfg: ExperimentConfig) -> None:
    start: float = time.time()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Config: {asdict(cfg)}")
    print(f"Device: {device}, torch {torch.__version__}")

    # ---------------------------------------------------------------- data
    stats_transform = transforms.Compose([
        transforms.Resize(size=(cfg.img_size, cfg.img_size)),
        transforms.ToImage(),
        transforms.ToDtype(dtype=torch.float32, scale=True),
    ])
    mean, std = compute_mean_std(
        dataset=IntelData(root=DATA_DIR, split="train", transform=stats_transform),
    )
    print(f"MEAN: {mean}\nSTD : {std}")
    transforms_train, transforms_test = build_transforms(cfg=cfg, mean=mean, std=std)

    train_set = IntelData(root=DATA_DIR, split="train", mode="train", transform=transforms_train)
    unlabeled_set = IntelData(
        root=DATA_DIR,
        split="train",
        mode="test",
        transform=transforms_test,
        use_unlabel=True,
    )
    valid_set = IntelData(root=DATA_DIR, split="val", mode="train", transform=transforms_test)
    test_set = IntelData(root=DATA_DIR, split="test", mode="test", transform=transforms_test)
    print(f"train {len(train_set)} | unlabeled {len(unlabeled_set)} | val {len(valid_set)}")

    loader_kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = DataLoader(dataset=train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(dataset=valid_set, shuffle=False, **loader_kwargs)

    # ---------------------------------------------------------------- model
    model = YourCNNModel(cfg=cfg, num_classes=len(CLASS_NAMES)).to(device)
    num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params / 1e6:.2f}M")

    # ----------------------------------------------------------- supervised
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = optim.AdamW(
        params=build_param_groups(model=model, weight_decay=cfg.weight_decay),
        lr=cfg.sup_lr,
        betas=(0.9, 0.999),
    )
    best_sup_acc: float = -1.0
    best_sup_state: dict[str, torch.Tensor] = snapshot(model)
    for epoch in range(cfg.sup_epochs):
        lr: float = warmup_cosine_lr(
            epoch=epoch,
            total_epochs=cfg.sup_epochs,
            warmup_epochs=cfg.sup_warmup_epochs,
            base_lr=cfg.sup_lr,
            min_lr=cfg.sup_min_lr,
        )
        set_lr(optimizer=optimizer, lr=lr)
        train_acc, train_loss = train_one_epoch(
            loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_acc, val_loss = evaluate(
            loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
        )
        mark: str = ""
        if val_acc > best_sup_acc:
            best_sup_acc = val_acc
            best_sup_state = snapshot(model)
            mark = " *"
        print(
            f"[SUP {epoch + 1:03d}/{cfg.sup_epochs}] lr {lr:.6f} | "
            f"train {train_acc:.4f}/{train_loss:.4f} | val {val_acc:.4f}/{val_loss:.4f}{mark}"
        )
    print(f"Best supervised val acc: {best_sup_acc:.4f}")

    # -------------------------------------------------------- self-training
    model.load_state_dict(best_sup_state)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.ssl_label_smoothing)
    optimizer = optim.AdamW(
        params=build_param_groups(model=model, weight_decay=cfg.ssl_weight_decay),
        lr=cfg.ssl_lr,
        betas=(0.9, 0.999),
    )
    best_acc, _ = evaluate(loader=val_loader, model=model, criterion=criterion, device=device)
    best_state: dict[str, torch.Tensor] = snapshot(model)
    unlabeled_indices: list[int] = list(range(len(unlabeled_set)))
    all_pseudo_datasets: list[Dataset] = []
    total_pseudo: int = 0
    train_loader_ssl: DataLoader = train_loader
    current_train_dataset: Dataset = train_set

    for epoch in range(cfg.ssl_epochs):
        if epoch % cfg.pseudo_interval == 0:
            pseudo_dataset, taken = get_pseudo_labels(
                model=model,
                unlabeled_set=unlabeled_set,
                unlabeled_indices=unlabeled_indices,
                threshold=cfg.threshold,
                transform=transforms_train,
                loader_kwargs=loader_kwargs,
                device=device,
            )
            print(f"Labeled {taken} images. Remaining unlabeled: {len(unlabeled_indices)}")
            if taken > 0:
                all_pseudo_datasets.append(pseudo_dataset)
                total_pseudo += taken
                current_train_dataset = ConcatDataset(
                    datasets=[train_set, *all_pseudo_datasets],
                )
                train_loader_ssl = DataLoader(
                    dataset=current_train_dataset,
                    shuffle=True,
                    **loader_kwargs,
                )

        lr = warmup_cosine_lr(
            epoch=epoch,
            total_epochs=cfg.ssl_epochs,
            warmup_epochs=cfg.ssl_warmup_epochs,
            base_lr=cfg.ssl_lr,
            min_lr=cfg.ssl_min_lr,
        )
        set_lr(optimizer=optimizer, lr=lr)
        train_acc, train_loss = train_one_epoch(
            loader=train_loader_ssl,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_acc, val_loss = evaluate(
            loader=val_loader,
            model=model,
            criterion=criterion,
            device=device,
        )
        mark = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = snapshot(model)
            mark = " *"
        print(
            f"[SSL {epoch + 1:03d}/{cfg.ssl_epochs}] lr {lr:.6f} | "
            f"train {train_acc:.4f}/{train_loss:.4f} | val {val_acc:.4f}/{val_loss:.4f} | "
            f"data {len(current_train_dataset)} (pseudo {total_pseudo}){mark}"
        )

    print(f"Best supervised val acc: {best_sup_acc:.4f}")
    print(f"Best self-training val acc: {best_acc:.4f}")
    print(f"Pseudo-labeled: {total_pseudo} / {len(unlabeled_set)}")

    # --------------------------------------------------------------- output
    is_record: bool = best_acc > CURRENT_BEST_VAL
    print(f"Current best: {CURRENT_BEST_VAL:.4f} -> new record: {is_record}")
    if WRITE_OUTPUTS and is_record:
        assert not CSV_PATH.exists() and not CKPT_PATH.exists(), "refuse to overwrite"
        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, CKPT_PATH)

        model.load_state_dict(best_state)
        model.eval()
        test_loader = DataLoader(dataset=test_set, shuffle=False, **loader_kwargs)
        predictions: list[int] = []
        with torch.no_grad():
            for images in test_loader:
                outputs = model(images.to(device, non_blocking=True))
                predictions.extend(outputs.argmax(dim=1).cpu().tolist())
        with CSV_PATH.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["ID", "label"])
            writer.writeheader()
            for path, pred in zip(test_set.paths, predictions):
                writer.writerow({"ID": path.name, "label": CLASS_NAMES[pred]})
        print(f"Saved {CSV_PATH.name} and {CKPT_PATH.name}")

    print(f"RESULT {EXPERIMENT_NAME} sup={best_sup_acc:.4f} val={best_acc:.4f}")
    print(f"Elapsed: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    assert not LOG_PATH.exists(), f"{LOG_PATH} already exists, refuse to overwrite"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w") as log_file:
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, log_file)
        main(CONFIG)
