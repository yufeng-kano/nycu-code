"""Validation AP and Kaggle CSV for the anchor-free detector (original-image coordinates)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .config import CLASS_NAMES
from .det_data import boxes_to_original
from .evaluate import coco_bbox_ap


def detections_to_rows(detection: Tensor | None, metadata: dict) -> list[tuple[list[float], float, int]]:
    """(K, 6) [x1, y1, x2, y2, score, class] on the network input -> original-coordinate rows with area > 0."""
    if detection is None or len(detection) == 0:
        return []
    boxes = boxes_to_original(detection[:, :4], metadata)
    rows = []
    for box, score, label in zip(boxes.tolist(), detection[:, 4].tolist(), detection[:, 5].tolist()):
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(([x1, y1, x2, y2], score, int(label)))
    return rows


@torch.inference_mode()
def evaluate_detector(
    model: nn.Module,
    loader: DataLoader,
    annotation_data: dict,
    category_ids: tuple[int, ...],
    conf_thres: float = 0.001,
    nms_thres: float = 0.65,
    hflip_tta: bool = False,
    tta_scales: tuple[float, ...] = (),
    fusion: str = "nms",
) -> dict:
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    for images, _, metadata in loader:
        images = images.to(device, non_blocking=True)
        detections = inference_tta(model, images, conf_thres, nms_thres, hflip=hflip_tta, scales=tta_scales, fusion=fusion)
        for detection, record in zip(detections, metadata):
            predictions.extend(detections_to_coco(detection, record, category_ids))
    return coco_bbox_ap(predictions, annotation_data, category_ids)


def detections_to_coco(detection: Tensor | None, metadata: dict, category_ids: tuple[int, ...]) -> list[dict]:
    """COCO result dicts (xywh in original pixels) for one image."""
    results = []
    for box, score, label in detections_to_rows(detection, metadata):
        x1, y1, x2, y2 = box
        results.append(
            {"image_id": metadata["image_id"], "category_id": category_ids[label], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score}
        )
    return results


def inference_tta(
    model: nn.Module,
    images: Tensor,
    conf_thres: float,
    nms_thres: float,
    hflip: bool = False,
    scales: tuple[float, ...] = (),
    max_detections: int = 100,
    fusion: str = "nms",
) -> list[Tensor | None]:
    """Test-time augmentation: the batch as is, optionally mirrored, optionally resized by each factor
    in `scales` (boxes scaled back), everything pooled per image and reduced with one class-aware NMS
    (`fusion="nms"`) or weighted boxes fusion over the variants (`fusion="wbf"`)."""
    per_image = tta_detections(model, images, conf_thres, nms_thres, hflip=hflip, scales=scales)
    n_members = (1 + len(scales)) * (2 if hflip else 1)
    return [
        fuse_detections(parts, fusion=fusion, iou_thres=nms_thres if fusion == "nms" else WBF_IOU, n_members=n_members, max_detections=max_detections)
        for parts in per_image
    ]


WBF_IOU: float = 0.55  # cluster threshold of weighted boxes fusion (Solovyev et al., 2021)


def tta_detections(
    model: nn.Module,
    images: Tensor,
    conf_thres: float,
    nms_thres: float,
    hflip: bool = False,
    scales: tuple[float, ...] = (),
) -> list[list[Tensor]]:
    """Per image, one (K, 6) [x1, y1, x2, y2, score, class] tensor per TTA variant, in the coordinates of `images`."""
    width = images.shape[-1]
    variants: list[tuple[Tensor, float, bool]] = [(images, 1.0, False)]
    if hflip:
        variants.append((images.flip(-1), 1.0, True))
    for scale in scales:
        size = int(round(images.shape[-1] * scale / 32)) * 32
        resized = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
        variants.append((resized, width / size, False))
        if hflip:
            variants.append((resized.flip(-1), width / size, True))

    pooled: list[list[Tensor]] = [[] for _ in range(len(images))]
    for batch, factor, flipped in variants:
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=images.device.type == "cuda"):
            outputs = model.inference(batch, conf_thres=conf_thres, nms_thres=nms_thres)
        for index, detection in enumerate(outputs):
            if detection is None:
                continue
            detection = detection.clone()
            if flipped:
                flipped_width = batch.shape[-1]
                detection[:, [0, 2]] = flipped_width - detection[:, [2, 0]]
            detection[:, :4] *= factor
            pooled[index].append(detection)
    return pooled


def fuse_detections(
    parts: list[Tensor],
    fusion: str,
    iou_thres: float,
    n_members: int,
    max_detections: int = 100,
) -> Tensor | None:
    """Reduce the detections of several members (TTA variants and / or models) of one image to at most
    `max_detections` boxes: class-aware NMS or weighted boxes fusion."""
    from torchvision.ops import batched_nms

    parts = [part for part in parts if part is not None and len(part)]
    if not parts:
        return None
    stacked = torch.cat(parts, dim=0)
    if fusion == "nms":
        keep = batched_nms(stacked[:, :4], stacked[:, 4], stacked[:, 5].long(), iou_thres)[:max_detections]
        return stacked[keep]
    if fusion == "wbf":
        fused = weighted_boxes_fusion(stacked, n_members=n_members, iou_thres=iou_thres)
        return fused[:max_detections]
    raise ValueError(f"unknown fusion {fusion!r}")


def weighted_boxes_fusion(detections: Tensor, n_members: int, iou_thres: float = WBF_IOU) -> Tensor:
    """Weighted boxes fusion (Solovyev, Wang, Gabruseva 2021) of (N, 6) [x1, y1, x2, y2, score, class]
    boxes coming from `n_members` members. Per class, boxes are visited by descending score and joined
    to the cluster whose fused box overlaps them by IoU > `iou_thres` (else they start a cluster); a
    cluster's box is the score-weighted mean of its members, its score the mean member score scaled
    by min(cluster size, n_members) / n_members. Returns (M, 6) sorted by descending score."""
    import numpy as np

    array = detections.detach().float().cpu().numpy()
    fused_rows: list[np.ndarray] = []
    for label in np.unique(array[:, 5]):
        boxes = array[array[:, 5] == label]
        boxes = boxes[np.argsort(-boxes[:, 4])]
        fused: list[np.ndarray] = []  # current fused xyxy per cluster
        weighted_sum: list[np.ndarray] = []  # sum(score * xyxy)
        score_sum: list[float] = []
        counts: list[int] = []
        for box in boxes:
            x1, y1, x2, y2, score = box[:5]
            target = -1
            if fused:
                stack = np.stack(fused)
                inter_w = np.clip(np.minimum(stack[:, 2], x2) - np.maximum(stack[:, 0], x1), 0, None)
                inter_h = np.clip(np.minimum(stack[:, 3], y2) - np.maximum(stack[:, 1], y1), 0, None)
                inter = inter_w * inter_h
                union = (stack[:, 2] - stack[:, 0]) * (stack[:, 3] - stack[:, 1]) + (x2 - x1) * (y2 - y1) - inter
                ious = inter / np.maximum(union, 1e-9)
                best = int(np.argmax(ious))
                if ious[best] > iou_thres:
                    target = best
            if target < 0:
                fused.append(box[:4].copy())
                weighted_sum.append(score * box[:4])
                score_sum.append(float(score))
                counts.append(1)
            else:
                weighted_sum[target] = weighted_sum[target] + score * box[:4]
                score_sum[target] += float(score)
                counts[target] += 1
                fused[target] = weighted_sum[target] / score_sum[target]
        for coords, total, count in zip(fused, score_sum, counts):
            confidence = total / count * min(count, n_members) / n_members
            fused_rows.append(np.array([*coords, confidence, label], dtype=np.float32))
    if not fused_rows:
        return detections[:0]
    result = np.stack(fused_rows)
    result = result[np.argsort(-result[:, 4])]
    return torch.from_numpy(result).to(detections.device)


@torch.inference_mode()
def write_submission(
    model: nn.Module,
    loader: DataLoader,
    output_path: Path,
    conf_thres: float = 0.001,
    nms_thres: float = 0.65,
    hflip_tta: bool = False,
    tta_scales: tuple[float, ...] = (),
    fusion: str = "nms",
) -> Path:
    """result.csv rows: id, JSON list of [class_name, confidence, x1, y1, x2, y2] (at most 100 per image)."""
    model.eval()
    device = next(model.parameters()).device
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    written = 0
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "prediction_list"])
        for images, _, metadata in loader:
            images = images.to(device, non_blocking=True)
            detections = inference_tta(model, images, conf_thres, nms_thres, hflip=hflip_tta, scales=tta_scales, fusion=fusion)
            for detection, record in zip(detections, metadata):
                rows = [
                    [CLASS_NAMES[label], score, *box]
                    for box, score, label in detections_to_rows(detection, record)
                ][:100]
                writer.writerow([record["file_name"], json.dumps(rows, allow_nan=False)])
                written += 1
    temporary.replace(output_path)
    print(f"Wrote {written:,} images to {output_path}")
    return output_path
