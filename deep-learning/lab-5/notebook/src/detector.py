"""Anchor-free detector: timm ImageNet-1k backbone -> PAFPN neck -> decoupled head with DFL box regression.

The head predicts, for every feature-map cell at strides 8/16/32, class logits and a discrete
distribution over the four distances (left, top, right, bottom) from the cell centre to the box
edges (Distribution Focal Loss parameterisation, `reg_max` bins per side, in units of the stride).
Training targets come from the task-aligned assigner in `det_loss.py`; inference decodes the
distributions, scores every (cell, class) pair and runs class-aware NMS.
"""

from __future__ import annotations

import math

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import batched_nms

from .config import NUM_CLASSES

STRIDES: tuple[int, int, int] = (8, 16, 32)


# ============================================================================ #
#                                 Building blocks                              #
# ============================================================================ #
class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 1, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel=3)
        self.conv2 = ConvBNAct(channels, channels, kernel=3)
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.shortcut else y


class CSPBlock(nn.Module):
    """Cross-stage partial block (C2f style): split, chain n bottlenecks, concatenate every stage, fuse."""

    def __init__(self, in_ch: int, out_ch: int, n: int, shortcut: bool) -> None:
        super().__init__()
        self.hidden = out_ch // 2
        self.conv_in = ConvBNAct(in_ch, 2 * self.hidden, kernel=1)
        self.blocks = nn.ModuleList(Bottleneck(self.hidden, shortcut) for _ in range(n))
        self.conv_out = ConvBNAct((2 + n) * self.hidden, out_ch, kernel=1)

    def forward(self, x: Tensor) -> Tensor:
        parts = list(self.conv_in(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.conv_out(torch.cat(parts, dim=1))


# ============================================================================ #
#                                    Backbone                                  #
# ============================================================================ #
class TimmBackbone(nn.Module):
    """Feature maps at strides 8, 16, 32 from a timm model (ImageNet-1k weights only)."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        drop_path_rate: float = 0.0,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            drop_path_rate=drop_path_rate,
        )
        if grad_checkpointing:
            self.body.set_grad_checkpointing(True)
        reductions = self.body.feature_info.reduction()
        self.indices = [reductions.index(stride) for stride in STRIDES]
        self.channels = tuple(self.body.feature_info.channels()[i] for i in self.indices)

    def forward(self, x: Tensor) -> list[Tensor]:
        features = self.body(x)
        return [features[i] for i in self.indices]


# ============================================================================ #
#                                      Neck                                    #
# ============================================================================ #
class PAFPN(nn.Module):
    """Top-down FPN followed by a bottom-up path; every output level has `width` channels."""

    def __init__(self, in_channels: tuple[int, int, int], width: int, depth: int) -> None:
        super().__init__()
        c3, c4, c5 = in_channels
        self.lateral3 = ConvBNAct(c3, width, kernel=1)
        self.lateral4 = ConvBNAct(c4, width, kernel=1)
        self.lateral5 = ConvBNAct(c5, width, kernel=1)
        self.top_down4 = CSPBlock(2 * width, width, n=depth, shortcut=False)
        self.top_down3 = CSPBlock(2 * width, width, n=depth, shortcut=False)
        self.down3 = ConvBNAct(width, width, kernel=3, stride=2)
        self.bottom_up4 = CSPBlock(2 * width, width, n=depth, shortcut=False)
        self.down4 = ConvBNAct(width, width, kernel=3, stride=2)
        self.bottom_up5 = CSPBlock(2 * width, width, n=depth, shortcut=False)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        c3, c4, c5 = features
        p5 = self.lateral5(c5)
        p4 = self.top_down4(torch.cat([F.interpolate(p5, scale_factor=2.0, mode="nearest"), self.lateral4(c4)], dim=1))
        p3 = self.top_down3(torch.cat([F.interpolate(p4, scale_factor=2.0, mode="nearest"), self.lateral3(c3)], dim=1))
        n4 = self.bottom_up4(torch.cat([self.down3(p3), p4], dim=1))
        n5 = self.bottom_up5(torch.cat([self.down4(n4), p5], dim=1))
        return [p3, n4, n5]


# ============================================================================ #
#                                      Head                                    #
# ============================================================================ #
class DecoupledHead(nn.Module):
    """Per level: separate classification and regression towers; regression emits 4 * reg_max DFL logits."""

    def __init__(self, width: int, num_classes: int, reg_max: int, image_size: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.cls_towers = nn.ModuleList()
        self.reg_towers = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        for stride in STRIDES:
            self.cls_towers.append(nn.Sequential(ConvBNAct(width, width, kernel=3), ConvBNAct(width, width, kernel=3)))
            self.reg_towers.append(nn.Sequential(ConvBNAct(width, width, kernel=3), ConvBNAct(width, width, kernel=3)))
            cls_pred = nn.Conv2d(width, num_classes, 1)
            reg_pred = nn.Conv2d(width, 4 * reg_max, 1)
            # Prior: about 5 objects per image spread over the cells of this level.
            nn.init.constant_(cls_pred.bias, math.log(5 / num_classes / (image_size / stride) ** 2))
            nn.init.constant_(reg_pred.bias, 1.0)
            self.cls_preds.append(cls_pred)
            self.reg_preds.append(reg_pred)

    def forward(self, features: list[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        cls_outputs, reg_outputs = [], []
        for i, feature in enumerate(features):
            cls_outputs.append(self.cls_preds[i](self.cls_towers[i](feature)))
            reg_outputs.append(self.reg_preds[i](self.reg_towers[i](feature)))
        return cls_outputs, reg_outputs


# ============================================================================ #
#                                    Detector                                  #
# ============================================================================ #
def make_anchors(feature_shapes: list[tuple[int, int]], device: torch.device) -> tuple[Tensor, Tensor]:
    """Cell-centre points in input pixels (A, 2) and their strides (A, 1), concatenated over levels."""
    points, strides = [], []
    for (height, width), stride in zip(feature_shapes, STRIDES):
        ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
        xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        points.append(torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))
        strides.append(torch.full((height * width, 1), float(stride), device=device))
    return torch.cat(points, dim=0), torch.cat(strides, dim=0)


def distribution_to_distance(reg_logits: Tensor, reg_max: int) -> Tensor:
    """(..., 4 * reg_max) DFL logits -> (..., 4) expected distances in stride units."""
    shape = reg_logits.shape[:-1]
    probs = reg_logits.reshape(*shape, 4, reg_max).softmax(dim=-1)
    bins = torch.arange(reg_max, device=reg_logits.device, dtype=probs.dtype)
    return (probs * bins).sum(dim=-1)


def distance_to_box(points: Tensor, distances: Tensor) -> Tensor:
    """Cell centres (A, 2) + ltrb distances (..., A, 4) in pixels -> xyxy boxes."""
    x1y1 = points - distances[..., :2]
    x2y2 = points + distances[..., 2:]
    return torch.cat([x1y1, x2y2], dim=-1)


class Detector(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        width: int = 256,
        depth: int = 2,
        reg_max: int = 16,
        image_size: int = 640,
        drop_path_rate: float = 0.0,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.backbone = TimmBackbone(
            model_name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            grad_checkpointing=grad_checkpointing,
        )
        self.neck = PAFPN(self.backbone.channels, width=width, depth=depth)
        self.head = DecoupledHead(width, num_classes, reg_max, image_size)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Raw outputs flattened over levels: cls logits (B, A, C), reg logits (B, A, 4 * reg_max), anchors."""
        cls_outputs, reg_outputs = self.head(self.neck(self.backbone(x)))
        shapes = [tuple(output.shape[-2:]) for output in cls_outputs]
        points, strides = make_anchors(shapes, x.device)
        cls_logits = torch.cat([o.flatten(2).transpose(1, 2) for o in cls_outputs], dim=1)
        reg_logits = torch.cat([o.flatten(2).transpose(1, 2) for o in reg_outputs], dim=1)
        return {"cls": cls_logits, "reg": reg_logits, "points": points, "strides": strides}

    def decode_boxes(self, outputs: dict[str, Tensor]) -> Tensor:
        """(B, A, 4) xyxy boxes in input pixels from the DFL logits (float32)."""
        distances = distribution_to_distance(outputs["reg"].float(), self.reg_max) * outputs["strides"]
        return distance_to_box(outputs["points"], distances)

    @torch.inference_mode()
    def inference(
        self,
        x: Tensor,
        conf_thres: float = 0.001,
        nms_thres: float = 0.65,
        max_detections: int = 100,
        pre_nms_top_k: int = 3000,
    ) -> list[Tensor | None]:
        """Per image an (K, 6) tensor [x1, y1, x2, y2, score, class] in input pixels, or None."""
        outputs = self(x)
        boxes = self.decode_boxes(outputs)
        scores = outputs["cls"].float().sigmoid()
        results: list[Tensor | None] = []
        for image_boxes, image_scores in zip(boxes, scores):
            # Multi-label: every (cell, class) pair above the threshold is a candidate.
            candidates = (image_scores > conf_thres).nonzero(as_tuple=False)
            if len(candidates) == 0:
                results.append(None)
                continue
            anchor_idx, class_idx = candidates[:, 0], candidates[:, 1]
            candidate_scores = image_scores[anchor_idx, class_idx]
            if len(candidate_scores) > pre_nms_top_k:
                candidate_scores, order = candidate_scores.topk(pre_nms_top_k)
                anchor_idx, class_idx = anchor_idx[order], class_idx[order]
            candidate_boxes = image_boxes[anchor_idx]
            keep = batched_nms(candidate_boxes, candidate_scores, class_idx, nms_thres)[:max_detections]
            results.append(
                torch.cat(
                    [candidate_boxes[keep], candidate_scores[keep, None], class_idx[keep, None].float()],
                    dim=1,
                )
            )
        return results
