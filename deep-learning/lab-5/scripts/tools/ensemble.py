"""Ensemble several finished runs at inference time (weighted boxes fusion or NMS over models x TTA variants).

Every run's best EMA weights are loaded from data/checkpoints/<run>.pt (model config from <run>-last.pt);
all models see the same letterbox input (--image-size), each produces its TTA variants (hflip, extra scales)
and all variants of all models are fused per image. Val AP is printed; with --csv NAME the test predictions
go to result/NAME.csv (never submitted automatically).

    python scripts/tools/ensemble.py --runs 011-convnext-b-40ep 010-convnext-b --image-size 896 --tta-scales 0.75 1.1
    python scripts/tools/ensemble.py --runs 011-convnext-b-40ep --fusion wbf --image-size 896 --tta-scales 0.75 1.1   (WBF for TTA pooling of one model)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

LAB_DIR: Path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB_DIR / "notebook"))

from src.config import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from src.det_data import DetectionDataset, detection_collate  # noqa: E402
from src.det_eval import WBF_IOU, detections_to_coco, detections_to_rows, fuse_detections, tta_detections  # noqa: E402
from src.detector import Detector  # noqa: E402
from src.evaluate import coco_bbox_ap  # noqa: E402

DATA_ROOT: Path = LAB_DIR / "data" / "dataset"
CKPT_DIR: Path = LAB_DIR / "data" / "checkpoints"


def load_run(run: str, image_size: int, device: torch.device) -> tuple[nn.Module, dict]:
    last = torch.load(CKPT_DIR / f"{run}-last.pt", map_location="cpu", weights_only=False)
    cfg = dict(last["config"])
    model = Detector(
        model_name=cfg["model_name"],
        num_classes=NUM_CLASSES,
        pretrained=False,
        width=cfg["width"],
        depth=cfg["depth"],
        reg_max=cfg["reg_max"],
        image_size=image_size,
    )
    model.load_state_dict(torch.load(CKPT_DIR / f"{run}.pt", map_location="cpu", weights_only=True))
    print(f"[{run}] {cfg['model_name']} | best val AP {last['best_ap']:.4f} at epoch {last['best_ap_epoch']}", flush=True)
    return model.to(device).eval(), cfg


@torch.inference_mode()
def ensemble_batch(
    models: list[nn.Module],
    images: torch.Tensor,
    conf_thres: float,
    nms_thres: float,
    hflip: bool,
    scales: tuple[float, ...],
    fusion: str,
    iou_thres: float,
) -> list[torch.Tensor | None]:
    per_image: list[list[torch.Tensor]] = [[] for _ in range(len(images))]
    for model in models:
        for index, parts in enumerate(tta_detections(model, images, conf_thres, nms_thres, hflip=hflip, scales=scales)):
            per_image[index].extend(parts)
    n_members = len(models) * (1 + len(scales)) * (2 if hflip else 1)
    return [fuse_detections(parts, fusion=fusion, iou_thres=iou_thres, n_members=n_members) for parts in per_image]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--tta-scales", type=float, nargs="*", default=[0.75, 1.1])
    parser.add_argument("--no-hflip", action="store_true")
    parser.add_argument("--fusion", choices=("wbf", "nms"), default="wbf")
    parser.add_argument("--iou", type=float, default=None, help="fusion IoU; default 0.55 for wbf, 0.65 for nms")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="evaluate the first N val images only (smoke)")
    parser.add_argument("--csv", default=None, help="also write result/<NAME>.csv for the test set")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    iou_thres = args.iou if args.iou is not None else (WBF_IOU if args.fusion == "wbf" else 0.65)
    scales = tuple(args.tta_scales)
    hflip = not args.no_hflip
    models = [load_run(run, args.image_size, device)[0] for run in args.runs]
    print(f"{len(models)} models x {(1 + len(scales)) * (2 if hflip else 1)} TTA variants, fusion {args.fusion} @ IoU {iou_thres}, input {args.image_size}", flush=True)

    def loader(list_file: str) -> tuple[DetectionDataset, DataLoader]:
        dataset = DetectionDataset(root=DATA_ROOT, list_file=DATA_ROOT / list_file, image_size=args.image_size, train=False)
        subset = torch.utils.data.Subset(dataset, range(args.limit)) if args.limit else dataset
        return dataset, DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=detection_collate, pin_memory=device.type == "cuda")

    val_dataset, val_loader = loader("val.txt")
    start = time.time()
    predictions: list[dict] = []
    image_ids: list[int] = []
    for images, _, metadata in val_loader:
        detections = ensemble_batch(models, images.to(device), args.conf, 0.65, hflip, scales, args.fusion, iou_thres)
        for detection, record in zip(detections, metadata):
            predictions.extend(detections_to_coco(detection, record, val_dataset.category_ids))
            image_ids.append(record["image_id"])
    annotation_data = val_dataset.coco
    if args.limit:
        keep = set(image_ids)
        annotation_data = {**annotation_data, "images": [i for i in annotation_data["images"] if i["id"] in keep], "annotations": [a for a in annotation_data["annotations"] if a["image_id"] in keep]}
    metrics = coco_bbox_ap(predictions, annotation_data, val_dataset.category_ids)
    print(f"val: AP {metrics['AP']:.4f} | AP50 {metrics['AP50']:.4f} | AP75 {metrics['AP75']:.4f} | APs {metrics['AP_small']:.4f} | APm {metrics['AP_medium']:.4f} | APl {metrics['AP_large']:.4f} | {(time.time() - start) / 60:.1f} min", flush=True)

    if not args.csv:
        return
    _, test_loader = loader("test.txt")
    output = LAB_DIR / "result" / f"{args.csv}.csv"
    temporary = output.with_suffix(".csv.tmp")
    written = 0
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "prediction_list"])
        for images, _, metadata in test_loader:
            detections = ensemble_batch(models, images.to(device), args.conf, 0.65, hflip, scales, args.fusion, iou_thres)
            for detection, record in zip(detections, metadata):
                rows = [[CLASS_NAMES[label], score, *box] for box, score, label in detections_to_rows(detection, record)][:100]
                writer.writerow([record["file_name"], json.dumps(rows, allow_nan=False)])
                written += 1
    temporary.replace(output)
    print(f"Wrote {written:,} images to {output}")


if __name__ == "__main__":
    main()
