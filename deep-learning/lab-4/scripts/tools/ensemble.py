"""Evaluate and ensemble finished mini-SAM runs: per-model and averaged-softmax val mIoU (hflip TTA), optional Kaggle CSV.

Usage:
    python scripts/tools/ensemble.py 002-mini-sam-g4 003-mini-sam-wide                      # val mIoU per model + ensemble
    python scripts/tools/ensemble.py 002-mini-sam-g4 003-mini-sam-wide --csv result/ens.csv  # also write the test CSV
    options: --weights 1 1.5 (per-model softmax weights), --no-tta, --batch 16, --sizes 320 384 448 (multi-scale
    inference by resizing the input and the decoder positional embedding), --min-area 50 (drop tiny predicted classes)

Each NAME loads checkpoints/NAME.pt ({"config", "state_dict", ...}) and rebuilds the model from scripts/train/NAME.py,
so the model code that produced a checkpoint is always the code used to run it. The validation protocol (per-image
mIoU over the classes present, 255 ignored) is the one implemented in the run scripts.
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
from torch.utils.data import DataLoader

LAB_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = LAB_DIR / "data"


def load_run(name: str, device: torch.device) -> tuple[ModuleType, torch.nn.Module]:
    script = LAB_DIR / "scripts" / "train" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"run_{name.replace('-', '_')}", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    ckpt = torch.load(LAB_DIR / "checkpoints" / f"{name}.pt", map_location="cpu", weights_only=False)
    cfg = module.ExperimentConfig(**ckpt["config"])
    model = module.MiniSAM(cfg=cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{name}: {params:.1f}M params, checkpoint val mIoU {ckpt.get('val_miou', float('nan')):.4f} (tta={ckpt.get('tta')})")
    return module, model


def set_decoder_grid(model: torch.nn.Module, size: int) -> None:
    """Resize the decoder's learned stride-16 positional embedding to the grid of a new input size."""
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
    decoder.image_pe = torch.nn.Parameter(pe.contiguous(), requires_grad=False)


@torch.no_grad()
def predict_probs(module: ModuleType, model: torch.nn.Module, images: torch.Tensor, sizes: list[int], tta: bool) -> torch.Tensor:
    """Softmax averaged over the given input sizes (bilinear resize in, logits resized back) and hflip."""
    h, w = images.shape[-2:]
    probs = None
    for size in sizes:
        x = images if size == h else F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
        set_decoder_grid(model=model, size=size)
        logits = module.predict_logits(model=model, images=x, tta_hflip=tta)
        if size != h:
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        p = F.softmax(logits, dim=1)
        probs = p if probs is None else probs + p
    set_decoder_grid(model=model, size=h)
    return probs / len(sizes)


def suppress_small_classes(probs: torch.Tensor, min_area: int) -> torch.Tensor:
    """Per image, zero the probability of every non-background class predicted on fewer than min_area pixels."""
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
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+")
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--csv", type=Path, default=None, help="write the Kaggle CSV of the ensemble to this path")
    parser.add_argument("--sizes", nargs="*", type=int, default=[320], help="input sizes for multi-scale inference (multiples of 64)")
    parser.add_argument("--min-area", type=int, default=0, help="drop predicted classes covering fewer pixels than this per image")
    args = parser.parse_args()
    weights = args.weights or [1.0] * len(args.names)
    assert len(weights) == len(args.names)
    device = torch.device("cuda")
    tta = not args.no_tta

    runs = [load_run(name=name, device=device) for name in args.names]
    ref = runs[0][0]  # datasets and metric helpers are identical across run scripts
    val_ids = ref.list_ids(folder=DATA_DIR / "val")
    val_set = ref.CIHPSegDataset(image_dir=DATA_DIR / "val", mask_dir=DATA_DIR / "val_mask", ids=val_ids, transform=None, hflip_prob=0.0)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch, shuffle=False, num_workers=4)

    start = time.time()
    per_model: list[list[torch.Tensor]] = [[] for _ in runs]
    ensemble: list[torch.Tensor] = []
    total_conf = torch.zeros(ref.N_CLASSES, ref.N_CLASSES, dtype=torch.long, device=device)
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        probs_sum = None
        for i, (module, model) in enumerate(runs):
            probs = predict_probs(module=module, model=model, images=images, sizes=args.sizes, tta=tta)
            per_model[i].append(ref.miou_from_confusion(conf=ref.confusion_per_image(pred=probs.argmax(dim=1), target=masks)))
            probs = probs * weights[i]
            probs_sum = probs if probs_sum is None else probs_sum + probs
        probs_sum = suppress_small_classes(probs=probs_sum, min_area=args.min_area)
        conf = ref.confusion_per_image(pred=probs_sum.argmax(dim=1), target=masks)
        ensemble.append(ref.miou_from_confusion(conf=conf))
        total_conf += conf.sum(dim=0)
    for name, scores in zip(args.names, per_model):
        print(f"val mIoU {name}: {torch.cat(scores).mean().item():.4f}")
    inter = total_conf.diagonal().float()
    union = total_conf.sum(dim=0).float() + total_conf.sum(dim=1).float() - inter
    per_class = (inter / union.clamp(min=1)).cpu()
    print(f"val mIoU ensemble ({', '.join(f'{n}x{w:g}' for n, w in zip(args.names, weights))}, sizes={args.sizes}, tta={tta}, min_area={args.min_area}): {torch.cat(ensemble).mean().item():.4f}")
    print("  per-class IoU: " + " ".join(f"{cls[:6]}={iou:.2f}" for cls, iou in zip(ref.CLASS_NAMES, per_class.tolist())))
    print(f"  ({time.time() - start:.0f}s)")

    if args.csv is None:
        return
    assert not args.csv.exists(), f"{args.csv} exists, refuse to overwrite"
    test_index = list(csv.DictReader((DATA_DIR / "test_index.csv").open(newline="")))
    test_ids = [row["filename"].removesuffix(".png") for row in test_index]
    assert [int(row["index"]) for row in test_index] == list(range(len(test_ids)))
    test_loader = DataLoader(dataset=ref.CIHPTestDataset(image_dir=DATA_DIR / "test", ids=test_ids), batch_size=args.batch, shuffle=False, num_workers=4)
    row_index = 0
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "pred_mask"])
        for images in test_loader:
            images = images.to(device)
            probs_sum = None
            for i, (module, model) in enumerate(runs):
                probs = predict_probs(module=module, model=model, images=images, sizes=args.sizes, tta=tta) * weights[i]
                probs_sum = probs if probs_sum is None else probs_sum + probs
            probs_sum = suppress_small_classes(probs=probs_sum, min_area=args.min_area)
            for pred in probs_sum.argmax(dim=1).cpu().numpy().astype(np.uint8):
                writer.writerow([row_index, json.dumps(pred.tolist(), separators=(",", ":"))])
                row_index += 1
    assert row_index == len(test_ids)
    print(f"wrote {args.csv} ({row_index} rows, {args.csv.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
