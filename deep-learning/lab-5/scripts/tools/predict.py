"""Evaluate a finished (or stopped) main-line run from its best checkpoint and write its Kaggle CSV.

Loads the experiment config and history from data/checkpoints/<run>-last.pt, the best EMA weights
from data/checkpoints/<run>.pt, reports val AP with and without hflip TTA, writes result/<run>.json
(config + per-epoch history + final metrics) and result/<run>.csv (test predictions, never submitted
automatically).

    python scripts/tools/predict.py --run 002-tal-convnext-t [--no-tta] [--batch-size 32]
    python scripts/tools/predict.py --run 004-mixup --image-size 768 --no-csv   (val AP at another input size only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

LAB_DIR: Path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LAB_DIR / "notebook"))

from src.config import NUM_CLASSES  # noqa: E402
from src.det_data import DetectionDataset, detection_collate  # noqa: E402
from src.det_eval import evaluate_detector, write_submission  # noqa: E402
from src.detector import Detector  # noqa: E402

DATA_ROOT: Path = LAB_DIR / "data" / "dataset"


def make_loader(dataset: DetectionDataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate,
        pin_memory=True,
    )


def main(
    run: str,
    batch_size: int,
    num_workers: int,
    tta: bool,
    image_size: int | None,
    write_csv: bool,
    tta_scales: tuple[float, ...],
    list_file: Path | None = None,
) -> None:
    ckpt_dir = LAB_DIR / "data" / "checkpoints"
    last = torch.load(ckpt_dir / f"{run}-last.pt", map_location="cpu", weights_only=False)
    cfg = dict(last["config"])
    if image_size is not None:
        cfg["image_size"] = image_size
    print(f"[{run}] best val AP {last['best_ap']:.4f} at epoch {last['best_ap_epoch']} (trained {last['epoch']} epochs)")

    model = Detector(
        model_name=cfg["model_name"],
        num_classes=NUM_CLASSES,
        pretrained=False,
        width=cfg["width"],
        depth=cfg["depth"],
        reg_max=cfg["reg_max"],
        image_size=cfg["image_size"],
    )
    model.load_state_dict(torch.load(ckpt_dir / f"{run}.pt", map_location="cpu", weights_only=True))
    model = model.cuda().eval()

    val_dataset = DetectionDataset(root=DATA_ROOT, list_file=list_file or DATA_ROOT / "val.txt", image_size=cfg["image_size"], train=False)
    val_loader = make_loader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    final: dict[str, dict] = {}
    for name, flip in (("val", False), ("val_tta", True)):
        if flip and not tta:
            break
        metrics = evaluate_detector(
            model,
            val_loader,
            annotation_data=val_dataset.coco,
            category_ids=val_dataset.category_ids,
            conf_thres=cfg["conf_thres"],
            nms_thres=cfg["nms_thres"],
            hflip_tta=flip,
            tta_scales=tta_scales if flip else (),
        )
        final[name] = metrics
        print(f"{name}: AP {metrics['AP']:.4f} | AP50 {metrics['AP50']:.4f} | AP75 {metrics['AP75']:.4f} | APs {metrics['AP_small']:.4f}")

    if not write_csv:
        return
    result_dir = LAB_DIR / "result"
    result_dir.mkdir(exist_ok=True)
    (result_dir / f"{run}.json").write_text(json.dumps({"config": cfg, "history": last["history"], "final": final}, indent=1))

    test_dataset = DetectionDataset(root=DATA_ROOT, list_file=DATA_ROOT / "test.txt", image_size=cfg["image_size"], train=False)
    write_submission(
        model,
        make_loader(test_dataset, batch_size=batch_size, num_workers=num_workers),
        result_dir / f"{run}.csv",
        conf_thres=cfg["conf_thres"],
        nms_thres=cfg["nms_thres"],
        hflip_tta=tta,
        tta_scales=tta_scales,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--image-size", type=int, default=None, help="evaluate at this letterbox size instead of the training one")
    parser.add_argument("--no-csv", action="store_true", help="only print val AP; write no json / csv")
    parser.add_argument("--tta-scales", type=float, nargs="*", default=[], help="extra input scale factors pooled in the TTA pass, e.g. 0.8 1.2")
    parser.add_argument("--list-file", type=Path, default=None, help="evaluate on this image list instead of val.txt (e.g. the notebook hold-out)")
    args = parser.parse_args()
    main(
        run=args.run,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tta=not args.no_tta,
        image_size=args.image_size,
        write_csv=not args.no_csv,
        tta_scales=tuple(args.tta_scales),
        list_file=args.list_file,
    )
