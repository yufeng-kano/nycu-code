"""Predict the course test set with a trained checkpoint and write the Kaggle CSV (index, pred_mask).

Two model families are supported:
  --arch unet   : the notebook U-Net (state_dict saved by HW4_315834011.ipynb, e.g. checkpoints/Unet-CIHP-best.pt)
  --arch script : a MiniSAM experiment; --script points at the training script whose MiniSAM/ExperimentConfig
                  classes and checkpoint format ({"config", "state_dict", ...}) are reused

Examples:
    python scripts/tools/predict.py --arch unet --ckpt checkpoints/Unet-CIHP-best.pt --out output.csv
    python scripts/tools/predict.py --arch script --script scripts/train/002-mini-sam-g4.py \
        --ckpt checkpoints/002-mini-sam-g4.pt --out result/002-mini-sam-g4.csv --tta
Add --val to also report the validation mIoU (course protocol) of the same weights before predicting.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

LAB_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = LAB_DIR / "data"
N_CLASSES: int = 20
IGNORE_INDEX: int = 255
MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
FLIP_CHANNEL_PERM: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 17, 16, 19, 18]


# ----------------------------------------------------------------------------- notebook U-Net (same layer names)
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels=n_channels, out_channels=64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        self.down4 = Down(in_channels=512, out_channels=1024)
        self.up1 = Up(in_channels=1024, out_channels=512)
        self.up2 = Up(in_channels=512, out_channels=256)
        self.up3 = Up(in_channels=256, out_channels=128)
        self.up4 = Up(in_channels=128, out_channels=64)
        self.outc = OutConv(in_channels=64, out_channels=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ----------------------------------------------------------------------------- data
class ImageFolderDataset(Dataset):
    def __init__(self, image_dir: Path, ids: list[str], mask_dir: Path | None = None) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.ids = ids
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.ids[index]
        with Image.open(self.image_dir / f"{stem}.png") as handle:
            image = np.array(handle.convert("RGB"))
        tensor = (torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0) - self.mean) / self.std
        if self.mask_dir is None:
            return tensor, torch.zeros(1, dtype=torch.long)
        with Image.open(self.mask_dir / f"{stem}.png") as handle:
            mask = np.array(handle, dtype=np.uint8)
        return tensor, torch.from_numpy(mask).long()


def load_script(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("exp_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp_module"] = module
    spec.loader.exec_module(module)
    return module


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, bool]:
    """Returns (model in eval mode, whether forward returns a (logits, aux) tuple)."""
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.arch == "unet":
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        model = UNet()
        model.load_state_dict(state)
        return model.to(device).eval(), False
    module = load_script(path=args.script)
    cfg = module.ExperimentConfig(**payload["config"])
    model = module.MiniSAM(cfg=cfg)
    model.load_state_dict(payload["state_dict"])
    print(f"loaded {args.ckpt.name}: val_miou in checkpoint = {payload.get('val_miou')}, tta flag = {payload.get('tta')}")
    return model.to(device).eval(), True


@torch.no_grad()
def forward_logits(model: nn.Module, tuple_output: bool, images: torch.Tensor, tta: bool) -> torch.Tensor:
    def run(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
        return (out[0] if tuple_output else out).float()

    logits = run(images)
    if tta:
        flipped = run(images.flip(dims=[3])).flip(dims=[3])[:, FLIP_CHANNEL_PERM]
        logits = 0.5 * (logits + flipped)
    return logits


def per_image_miou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    index = target.clamp(max=N_CLASSES - 1) * N_CLASSES + pred
    index = torch.where(valid, index, torch.full_like(index, N_CLASSES * N_CLASSES))
    counts = torch.stack([torch.bincount(row.flatten(), minlength=N_CLASSES * N_CLASSES + 1) for row in index])
    conf = counts[:, : N_CLASSES * N_CLASSES].view(-1, N_CLASSES, N_CLASSES)
    inter = conf.diagonal(dim1=1, dim2=2).float()
    union = conf.sum(dim=1).float() + conf.sum(dim=2).float() - inter
    present = union > 0
    iou = torch.where(present, inter / union.clamp(min=1), torch.zeros_like(inter))
    return iou.sum(dim=1) / present.sum(dim=1).clamp(min=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=["unet", "script"], required=True)
    parser.add_argument("--script", type=Path, default=None, help="training script defining MiniSAM (arch=script)")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="CSV path (index,pred_mask)")
    parser.add_argument("--tta", action="store_true", help="average with the label-aware horizontal flip")
    parser.add_argument("--val", action="store_true", help="report val mIoU first")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="only the first N test images (format check)")
    args = parser.parse_args()
    if args.arch == "script" and args.script is None:
        parser.error("--script is required with --arch script")

    device = torch.device("cuda")
    model, tuple_output = build_model(args=args, device=device)

    if args.val:
        val_ids = sorted(p.stem for p in (DATA_DIR / "val").glob("*.png"))
        loader = DataLoader(
            dataset=ImageFolderDataset(image_dir=DATA_DIR / "val", ids=val_ids, mask_dir=DATA_DIR / "val_mask"),
            batch_size=args.batch_size,
            num_workers=4,
        )
        scores = []
        for images, masks in loader:
            logits = forward_logits(model=model, tuple_output=tuple_output, images=images.to(device), tta=args.tta)
            scores.append(per_image_miou(pred=logits.argmax(dim=1), target=masks.to(device)))
        print(f"val mIoU (per-image mean, tta={args.tta}): {torch.cat(scores).mean().item():.4f}")

    rows = list(csv.DictReader((DATA_DIR / "test_index.csv").open(newline="")))
    assert [int(r["index"]) for r in rows] == list(range(len(rows)))
    test_ids = [r["filename"].removesuffix(".png") for r in rows]
    if args.limit:
        test_ids = test_ids[: args.limit]
    loader = DataLoader(
        dataset=ImageFolderDataset(image_dir=DATA_DIR / "test", ids=test_ids),
        batch_size=args.batch_size,
        num_workers=4,
    )
    started = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    row_index = 0
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "pred_mask"])
        for images, _ in loader:
            logits = forward_logits(model=model, tuple_output=tuple_output, images=images.to(device), tta=args.tta)
            for pred in logits.argmax(dim=1).cpu().numpy().astype(np.uint8):
                writer.writerow([row_index, json.dumps(pred.tolist(), separators=(",", ":"))])
                row_index += 1
    print(f"wrote {args.out} ({row_index} rows, {args.out.stat().st_size / 1e9:.2f} GB) in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
