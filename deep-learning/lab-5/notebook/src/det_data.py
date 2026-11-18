"""Detection data pipeline for the anchor-free detector: letterbox, mosaic, affine, HSV, flip.

Boxes are kept as pixel xyxy on the network input. Unlike `dataset.py`, no grid targets are built here;
the assigner in `det_loss.py` matches anchors to boxes on the fly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

from .config import IMG_MEAN, IMG_STD
from .dataset import CocoDetectorDataset

PAD_VALUE: int = 114
cv2.setNumThreads(0)  # one thread per dataloader worker; otherwise every worker spawns a full thread pool


@dataclass
class AugmentConfig:
    mosaic: float = 1.0  # probability of mosaic per training sample
    mixup: float = 0.0  # probability of mixing a second (mosaic) sample in
    scale: tuple[float, float] = (0.5, 1.5)  # random affine scale range
    translate: float = 0.1  # random affine translation, fraction of the input size
    hflip: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    min_box_size: float = 2.0  # pixels; smaller boxes after augmentation are dropped
    min_area_ratio: float = 0.1  # boxes that lose more than 90% of their area are dropped
    copy_paste: float = 0.0  # probability of pasting object crops from other images onto the mosaic canvas
    copy_paste_objects: int = 4  # at most this many pasted objects per mosaic


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping the aspect ratio and pad to a size x size canvas; returns ratio and (dx, dy) padding."""
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    if (new_w, new_h) != (width, height):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dx, dy = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(dy - 0.1), round(dy + 0.1)
    left, right = round(dx - 0.1), round(dx + 0.1)
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    return image, ratio, (float(left), float(top))


def augment_hsv(image: np.ndarray, h_gain: float, s_gain: float, v_gain: float) -> np.ndarray:
    """Random HSV jitter in place on an RGB uint8 image."""
    if h_gain == 0 and s_gain == 0 and v_gain == 0:
        return image
    gains = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    table = np.arange(0, 256, dtype=gains.dtype)
    lut_hue = ((table * gains[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(table * gains[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(table * gains[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def random_affine(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    out_size: int,
    scale: tuple[float, float],
    translate: float,
    min_box_size: float,
    min_area_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random scale + translation about the canvas centre, output cropped to out_size x out_size."""
    height, width = image.shape[:2]
    # Move the canvas centre to the origin, scale, then move to the output centre plus a random shift.
    s = random.uniform(scale[0], scale[1])
    tx = out_size / 2 + random.uniform(-translate, translate) * out_size
    ty = out_size / 2 + random.uniform(-translate, translate) * out_size
    matrix = np.array(
        [[s, 0.0, tx - s * width / 2], [0.0, s, ty - s * height / 2]],
        dtype=np.float64,
    )
    image = cv2.warpAffine(
        image,
        matrix,
        dsize=(out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderValue=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    if len(boxes) == 0:
        return image, boxes, labels
    # Transform all four corners and take the enclosing box (pure scale + shift keeps boxes axis aligned).
    corners = np.stack(
        [boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]],
        axis=1,
    ).reshape(-1, 2)
    corners = corners * s + np.array([matrix[0, 2], matrix[1, 2]])
    corners = corners.reshape(-1, 4, 2)
    new_boxes = np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1)
    area_before = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
    new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, out_size)
    new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, out_size)
    wh = new_boxes[:, 2:] - new_boxes[:, :2]
    keep = (
        (wh[:, 0] > min_box_size)
        & (wh[:, 1] > min_box_size)
        & (wh[:, 0] * wh[:, 1] / (area_before + 1e-9) > min_area_ratio)
    )
    return image, new_boxes[keep].astype(np.float32), labels[keep]


class DetectionDataset(CocoDetectorDataset):
    """Letterboxed images with pixel-space boxes; mosaic / affine / HSV / flip augmentation when training."""

    def __init__(
        self,
        root: Path | str,
        list_file: Path | str,
        image_size: int,
        train: bool,
        augment: AugmentConfig | None = None,
    ) -> None:
        super().__init__(
            root=root,
            list_file=list_file,
            train=train,
            encode_target=False,
            image_size=image_size,
        )
        self.augment = augment if augment is not None else AugmentConfig()
        self.mosaic_enabled = train
        self.mean = np.array(IMG_MEAN, dtype=np.float32)
        self.std = np.array(IMG_STD, dtype=np.float32)

    # ------------------------------------------------------------------ loading
    def load_image(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
        """RGB image, pixel xyxy boxes (crowd boxes dropped), labels, original (width, height)."""
        record = self.images[index]
        image = cv2.imread(str(self.root / record["file_name"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(record["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        width, height = record["width"], record["height"]
        boxes = self.boxes[index] * np.array([width, height, width, height], dtype=np.float32)
        return image, boxes.astype(np.float32), self.labels[index].copy(), (width, height)

    def load_resized(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Image with its long side scaled to image_size (no padding), boxes scaled accordingly."""
        image, boxes, labels, (width, height) = self.load_image(index)
        ratio = self.image_size / max(width, height)
        if ratio != 1:
            image = cv2.resize(
                image,
                (round(width * ratio), round(height * ratio)),
                interpolation=cv2.INTER_LINEAR,
            )
            boxes = boxes * ratio
        return image, boxes, labels

    # ------------------------------------------------------------------ mosaic
    def load_mosaic(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Four images on a 2s x 2s canvas around a random centre (YOLOv5 style), then random affine to s x s."""
        s = self.image_size
        xc, yc = (int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2))
        indices = [index] + random.choices(range(len(self)), k=3)
        canvas = np.full((s * 2, s * 2, 3), PAD_VALUE, dtype=np.uint8)
        all_boxes: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for position, idx in enumerate(indices):
            image, boxes, labels = self.load_resized(idx)
            h, w = image.shape[:2]
            if position == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif position == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif position == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            shift = np.array([x1a - x1b, y1a - y1b, x1a - x1b, y1a - y1b], dtype=np.float32)
            all_boxes.append(boxes + shift)
            all_labels.append(labels)
        boxes = np.concatenate(all_boxes, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        boxes = boxes.clip(0, 2 * s)
        aug = self.augment
        if random.random() < aug.copy_paste:
            boxes, labels = self.copy_paste(canvas, boxes, labels, aug.copy_paste_objects)
        return random_affine(
            canvas,
            boxes,
            labels,
            out_size=s,
            scale=aug.scale,
            translate=aug.translate,
            min_box_size=aug.min_box_size,
            min_area_ratio=aug.min_area_ratio,
        )

    # ------------------------------------------------------------------ copy-paste
    def copy_paste(
        self,
        canvas: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        max_objects: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Box-level copy-paste (Ghiasi et al. 2021, without masks): crops of objects from random other images,
        pasted in place on the mosaic canvas where they cover little of the existing boxes. Crops are taken at the
        resized scale of their source image, so pasted objects keep a realistic size; each is flipped at random."""
        canvas_size = canvas.shape[0]
        new_boxes: list[np.ndarray] = []
        new_labels: list[int] = []
        for _ in range(random.randint(1, max_objects)):
            image, src_boxes, src_labels = self.load_resized(random.randrange(len(self)))
            if len(src_boxes) == 0:
                continue
            k = random.randrange(len(src_boxes))
            x1, y1, x2, y2 = (int(round(v)) for v in src_boxes[k])
            w, h = x2 - x1, y2 - y1
            if w < 8 or h < 8 or w * h > (canvas_size // 3) ** 2:
                continue
            crop = image[y1:y2, x1:x2]
            if random.random() < 0.5:
                crop = crop[:, ::-1]
            for _attempt in range(5):
                px, py = random.randint(0, canvas_size - w), random.randint(0, canvas_size - h)
                candidate = np.array([px, py, px + w, py + h], dtype=np.float32)
                occupied = np.concatenate([boxes, *new_boxes], axis=0) if new_boxes else boxes
                if len(occupied) == 0 or self.coverage(candidate, occupied).max() < 0.3:
                    canvas[py : py + h, px : px + w] = crop
                    new_boxes.append(candidate[None])
                    new_labels.append(int(src_labels[k]))
                    break
        if not new_boxes:
            return boxes, labels
        return (
            np.concatenate([boxes, *new_boxes], axis=0),
            np.concatenate([labels, np.array(new_labels, dtype=labels.dtype)], axis=0),
        )

    @staticmethod
    def coverage(box: np.ndarray, others: np.ndarray) -> np.ndarray:
        """Fraction of each box in `others` covered by `box`."""
        ix1 = np.maximum(box[0], others[:, 0])
        iy1 = np.maximum(box[1], others[:, 1])
        ix2 = np.minimum(box[2], others[:, 2])
        iy2 = np.minimum(box[3], others[:, 3])
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        area = np.clip((others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]), 1e-6, None)
        return inter / area

    # ------------------------------------------------------------------ samples
    def train_sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        aug = self.augment
        if self.mosaic_enabled and random.random() < aug.mosaic:
            image, boxes, labels = self.load_mosaic(index)
            if random.random() < aug.mixup:
                image2, boxes2, labels2 = self.load_mosaic(random.randrange(len(self)))
                lam = np.random.beta(32.0, 32.0)
                image = (image.astype(np.float32) * lam + image2.astype(np.float32) * (1 - lam)).astype(np.uint8)
                boxes = np.concatenate([boxes, boxes2], axis=0)
                labels = np.concatenate([labels, labels2], axis=0)
        else:
            image, boxes, labels, _ = self.load_image(index)
            image, ratio, (dx, dy) = letterbox(image, self.image_size)
            boxes = boxes * ratio + np.array([dx, dy, dx, dy], dtype=np.float32)
            image, boxes, labels = random_affine(
                image,
                boxes,
                labels,
                out_size=self.image_size,
                scale=aug.scale,
                translate=aug.translate,
                min_box_size=aug.min_box_size,
                min_area_ratio=aug.min_area_ratio,
            )
        image = augment_hsv(image, aug.hsv_h, aug.hsv_s, aug.hsv_v)
        if random.random() < aug.hflip:
            image = np.ascontiguousarray(image[:, ::-1])
            if len(boxes):
                boxes[:, [0, 2]] = self.image_size - boxes[:, [2, 0]]
        return image, boxes, labels

    def to_tensor(self, image: np.ndarray) -> Tensor:
        array = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict]:
        record = self.images[index]
        if self.train:
            image, boxes, labels = self.train_sample(index)
            ratio, pad = 1.0, (0.0, 0.0)
        else:
            image, boxes, labels, _ = self.load_image(index)
            image, ratio, pad = letterbox(image, self.image_size)
            boxes = boxes * ratio + np.array([pad[0], pad[1], pad[0], pad[1]], dtype=np.float32)
        target = torch.from_numpy(
            np.concatenate([labels[:, None].astype(np.float32), boxes.reshape(-1, 4)], axis=1)
        )
        metadata = {
            "image_id": record["id"],
            "file_name": Path(record["file_name"]).name,
            "original_size": (record["width"], record["height"]),
            "ratio": ratio,
            "pad": pad,
        }
        return self.to_tensor(image), target, metadata


def detection_collate(batch: list[tuple[Tensor, Tensor, dict]]) -> tuple[Tensor, Tensor, list[dict]]:
    """Stack images; targets become (M, 6) rows of [batch_index, class, x1, y1, x2, y2]."""
    images, targets, metadata = zip(*batch)
    rows = [
        torch.cat([torch.full((len(target), 1), float(i)), target], dim=1)
        for i, target in enumerate(targets)
    ]
    return torch.stack(images), torch.cat(rows, dim=0), list(metadata)


def boxes_to_original(boxes: Tensor, metadata: dict) -> Tensor:
    """Undo letterbox: network-input xyxy -> original-image xyxy, clipped to the image."""
    width, height = metadata["original_size"]
    dx, dy = metadata["pad"]
    out = (boxes - boxes.new_tensor([dx, dy, dx, dy])) / metadata["ratio"]
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0, height)
    return out
