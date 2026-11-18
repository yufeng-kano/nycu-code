"""Task-aligned label assignment and the detection loss (BCE classification, CIoU box, DFL).

Assignment (TOOD / YOLOv8 style): for every ground-truth box, anchors whose centre lies inside the box
are candidates; each candidate is scored by `alignment = score^alpha * IoU^beta`, the top-k aligned
candidates become positives, and an anchor claimed by several boxes goes to the one with the highest IoU.
Classification targets are the alignment metric rescaled so each box's best anchor gets its best IoU.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .detector import distance_to_box, distribution_to_distance


def box_iou_pairwise(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """IoU between (..., N, 4) and (..., M, 4) xyxy boxes -> (..., N, M)."""
    lt = torch.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    rb = torch.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    return inter / (area1[..., :, None] + area2[..., None, :] - inter + eps)


def ciou(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """Complete IoU between matched (N, 4) xyxy boxes -> (N,)."""
    lt = torch.maximum(pred[:, :2], target[:, :2])
    rb = torch.minimum(pred[:, 2:], target[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(dim=-1)
    wh1 = (pred[:, 2:] - pred[:, :2]).clamp(min=0)
    wh2 = (target[:, 2:] - target[:, :2]).clamp(min=0)
    union = wh1.prod(dim=-1) + wh2.prod(dim=-1) - inter + eps
    iou = inter / union
    enclose_wh = torch.maximum(pred[:, 2:], target[:, 2:]) - torch.minimum(pred[:, :2], target[:, :2])
    diagonal_sq = enclose_wh.pow(2).sum(dim=-1) + eps
    center_dist_sq = ((pred[:, :2] + pred[:, 2:]) - (target[:, :2] + target[:, 2:])).pow(2).sum(dim=-1) / 4
    v = (4 / math.pi**2) * (torch.atan(wh2[:, 0] / (wh2[:, 1] + eps)) - torch.atan(wh1[:, 0] / (wh1[:, 1] + eps))).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - center_dist_sq / diagonal_sq - alpha * v


# ============================================================================ #
#                              Task-aligned assigner                           #
# ============================================================================ #
class TaskAlignedAssigner(nn.Module):
    def __init__(self, top_k: int = 10, alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9) -> None:
        super().__init__()
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,  # (B, A, C) sigmoid probabilities
        pred_boxes: Tensor,  # (B, A, 4) xyxy pixels
        points: Tensor,  # (A, 2)
        gt_labels: Tensor,  # (B, G) long, padded
        gt_boxes: Tensor,  # (B, G, 4) xyxy pixels, padded
        gt_mask: Tensor,  # (B, G) bool, True for real boxes
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns target_boxes (B, A, 4), target_scores (B, A, C), fg_mask (B, A) bool, target_labels (B, A)."""
        batch, num_gt = gt_labels.shape
        num_anchors, num_classes = pred_scores.shape[1:]
        device = pred_scores.device
        if num_gt == 0:
            return (
                torch.zeros(batch, num_anchors, 4, device=device),
                torch.zeros(batch, num_anchors, num_classes, device=device),
                torch.zeros(batch, num_anchors, dtype=torch.bool, device=device),
                torch.zeros(batch, num_anchors, dtype=torch.long, device=device),
            )

        # Candidates: anchor centres inside the box.
        deltas = torch.cat(
            [points[None, None] - gt_boxes[:, :, None, :2], gt_boxes[:, :, None, 2:] - points[None, None]],
            dim=-1,
        )  # (B, G, A, 4)
        in_box = deltas.amin(dim=-1) > self.eps  # (B, G, A)

        # Alignment metric.
        overlaps = box_iou_pairwise(gt_boxes, pred_boxes)  # (B, G, A)
        batch_idx = torch.arange(batch, device=device)[:, None].expand(-1, num_gt)
        class_scores = pred_scores.permute(0, 2, 1)[batch_idx, gt_labels]  # (B, G, A): score of each box's class
        alignment = class_scores.pow(self.alpha) * overlaps.pow(self.beta)
        valid = in_box & gt_mask[:, :, None]
        alignment = alignment * valid

        # Top-k per box among candidates.
        top_values, top_indices = alignment.topk(self.top_k, dim=-1)
        top_mask = top_values > self.eps
        mask_pos = torch.zeros_like(alignment, dtype=torch.bool)
        mask_pos.scatter_(-1, top_indices, top_mask)
        mask_pos &= valid

        # Anchors claimed by several boxes go to the box with the highest IoU.
        fg_count = mask_pos.sum(dim=1)  # (B, A)
        if (fg_count > 1).any():
            multiple = (fg_count > 1)[:, None, :].expand(-1, num_gt, -1)
            best_gt = (overlaps * mask_pos).argmax(dim=1)  # (B, A)
            best_mask = F.one_hot(best_gt, num_gt).permute(0, 2, 1).bool()
            mask_pos = torch.where(multiple, best_mask, mask_pos)
            fg_count = mask_pos.sum(dim=1)
        fg_mask = fg_count > 0
        target_gt_idx = mask_pos.float().argmax(dim=1)  # (B, A)

        # Gather targets.
        flat_idx = target_gt_idx + torch.arange(batch, device=device)[:, None] * num_gt
        target_labels = gt_labels.reshape(-1)[flat_idx]
        target_boxes = gt_boxes.reshape(-1, 4)[flat_idx]
        target_scores = F.one_hot(target_labels, num_classes).float() * fg_mask[..., None]

        # Rescale: each box's best-aligned anchor is scored with that box's best IoU.
        alignment = alignment * mask_pos
        max_alignment = alignment.amax(dim=-1, keepdim=True)  # (B, G, 1)
        max_overlap = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        normalised = (alignment * max_overlap / (max_alignment + self.eps)).amax(dim=1)  # (B, A)
        target_scores = target_scores * normalised[..., None]
        return target_boxes, target_scores, fg_mask, target_labels


# ============================================================================ #
#                                  Detection loss                              #
# ============================================================================ #
class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        reg_max: int,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
        top_k: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        self.assigner = TaskAlignedAssigner(top_k=top_k, alpha=alpha, beta=beta)

    @staticmethod
    def pad_targets(targets: Tensor, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        """(M, 6) rows [batch, class, x1, y1, x2, y2] -> padded (B, G) labels, (B, G, 4) boxes, (B, G) mask."""
        counts = torch.bincount(targets[:, 0].long(), minlength=batch_size)
        num_gt = int(counts.max().item()) if len(targets) else 0
        labels = torch.zeros(batch_size, num_gt, dtype=torch.long, device=device)
        boxes = torch.zeros(batch_size, num_gt, 4, device=device)
        mask = torch.zeros(batch_size, num_gt, dtype=torch.bool, device=device)
        if num_gt:
            targets = targets.to(device)
            order = torch.argsort(targets[:, 0], stable=True)
            targets = targets[order]
            batch_idx = targets[:, 0].long()
            starts = torch.cumsum(counts, dim=0) - counts
            slot = torch.arange(len(targets), device=device) - starts[batch_idx]
            labels[batch_idx, slot] = targets[:, 1].long()
            boxes[batch_idx, slot] = targets[:, 2:6]
            mask[batch_idx, slot] = True
        return labels, boxes, mask

    def dfl_loss(self, reg_logits: Tensor, target_distances: Tensor) -> Tensor:
        """Cross-entropy against the two integer bins around each target distance, (N, 4) -> (N,)."""
        target_distances = target_distances.clamp(0, self.reg_max - 1 - 0.01)
        low = target_distances.long()
        high = low + 1
        weight_high = target_distances - low.float()
        weight_low = 1.0 - weight_high
        logits = reg_logits.reshape(-1, self.reg_max)
        loss = (
            F.cross_entropy(logits, low.reshape(-1), reduction="none") * weight_low.reshape(-1)
            + F.cross_entropy(logits, high.reshape(-1), reduction="none") * weight_high.reshape(-1)
        )
        return loss.reshape(-1, 4).mean(dim=-1)

    def forward(self, outputs: dict[str, Tensor], targets: Tensor) -> dict[str, Tensor]:
        cls_logits = outputs["cls"].float()
        reg_logits = outputs["reg"].float()
        points, strides = outputs["points"], outputs["strides"]
        batch_size = cls_logits.shape[0]
        device = cls_logits.device

        distances = distribution_to_distance(reg_logits, self.reg_max) * strides  # (B, A, 4) pixels
        pred_boxes = distance_to_box(points, distances)

        gt_labels, gt_boxes, gt_mask = self.pad_targets(targets, batch_size, device)
        target_boxes, target_scores, fg_mask, _ = self.assigner(
            cls_logits.detach().sigmoid(),
            pred_boxes.detach(),
            points,
            gt_labels,
            gt_boxes,
            gt_mask,
        )
        target_scores_sum = target_scores.sum().clamp(min=1.0)

        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, target_scores, reduction="sum") / target_scores_sum

        box_loss = cls_logits.new_zeros(())
        dfl = cls_logits.new_zeros(())
        if fg_mask.any():
            weight = target_scores.sum(dim=-1)[fg_mask]
            box_loss = ((1.0 - ciou(pred_boxes[fg_mask], target_boxes[fg_mask])) * weight).sum() / target_scores_sum
            fg_points = points[None].expand(batch_size, -1, -1)[fg_mask]
            fg_strides = strides[None].expand(batch_size, -1, -1)[fg_mask]
            target_ltrb = torch.cat(
                [fg_points - target_boxes[fg_mask][:, :2], target_boxes[fg_mask][:, 2:] - fg_points],
                dim=-1,
            ) / fg_strides
            dfl = (self.dfl_loss(reg_logits[fg_mask], target_ltrb) * weight).sum() / target_scores_sum

        total = self.box_weight * box_loss + self.cls_weight * cls_loss + self.dfl_weight * dfl
        return {"total": total, "box": box_loss, "cls": cls_loss, "dfl": dfl, "num_fg": fg_mask.sum().float()}
