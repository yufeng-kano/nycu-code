"""DL Lab 4 experiment 014-mini-sam-lovasz: 008 with the batch Dice term replaced by a per-image Lovasz-softmax loss.

Changes vs 008: loss = 0.5 CE + 0.5 Lovasz-softmax (Berman et al., CVPR 2018) computed per image over the classes present
in that image, which is the Lovasz extension of exactly the quantity the leaderboard averages (per-image mIoU over the
classes present); the batched implementation sorts all B x C error vectors at once. Everything else identical to 008;
resumable: the full training state is written to checkpoints/<name>-last.pt after every epoch and picked up
automatically on restart (Colab sessions get reclaimed), the log is then appended instead of refused.

Architecture (every weight random-initialised, no external data):
  conv stem (stride 4) -> 4-stage hierarchical transformer encoder (Hiera-like:
  windowed attention at strides 4/8, global attention at strides 16/32, depthwise-conv
  positional encoding, LayerScale, stochastic depth) -> FPN neck -> SAM-style two-way
  transformer mask decoder with 20 learnable class tokens whose hypernetwork outputs
  are dotted with a stride-4 pixel embedding -> 320 x 320 logits.
  Auxiliary per-pixel head on the stride-4 FPN feature for deep supervision.

Run from anywhere:  python scripts/train/002-mini-sam-g4.py            (full run or resume, writes logs/result/checkpoint)
                    python scripts/train/002-mini-sam-g4.py --smoke    (30 steps, prints img/s and memory only)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

# ============================================================================ #
#                                Experiment                                    #
# ============================================================================ #
EXPERIMENT_NAME: str = "014-mini-sam-lovasz"
CURRENT_BEST_VAL: float = 0.5989  # 008-mini-sam-wide-deep (val mIoU with hflip TTA)
WRITE_OUTPUTS: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 2026
    # data / augmentation
    img_size: int = 320
    scale_range: tuple[float, float] = (0.5, 2.0)
    translate: float = 0.2
    rotate_deg: float = 10.0
    affine_p: float = 0.8
    batch_size: int = 16
    num_workers: int = 16
    # encoder
    stem_dim: int = 48
    dims: tuple[int, ...] = (96, 192, 384, 768)
    depths: tuple[int, ...] = (2, 2, 12, 2)
    heads: tuple[int, ...] = (3, 6, 12, 24)
    windows: tuple[int, ...] = (8, 8, 0, 0)  # 0 = global attention
    stage_types: tuple[str, ...] = ("attn", "attn", "attn", "attn")  # "attn" or "conv" (ConvNeXt block)
    mlp_ratio: float = 4.0
    layer_scale_init: float = 0.1
    drop_path_rate: float = 0.2
    # neck + decoder
    neck_dim: int = 256
    decoder_depth: int = 2
    decoder_heads: int = 8
    decoder_mlp_dim: int = 1024
    mask_dim: int = 64
    aux_weight: float = 0.4
    # loss
    ce_weight: float = 0.5
    lovasz_weight: float = 0.5
    # optimisation
    epochs: int = 100
    warmup_epochs: int = 5
    lr: float = 6e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    tta_hflip: bool = True
    notes: list[str] = field(default_factory=list)


CONFIG = ExperimentConfig()

# ============================================================================ #
#                                  Paths                                       #
# ============================================================================ #
# On a Colab VM the script is sent by `colab exec -f`, so __file__ may be missing: LAB_DIR then comes from the environment.
LAB_DIR: Path = Path(os.environ["LAB_DIR"]).resolve() if "LAB_DIR" in os.environ else Path(__file__).resolve().parents[2]
DATA_DIR: Path = LAB_DIR / "data"
LOG_PATH: Path = LAB_DIR / "logs" / f"{EXPERIMENT_NAME}.log"
CSV_PATH: Path = LAB_DIR / "result" / f"{EXPERIMENT_NAME}.csv"
CKPT_PATH: Path = LAB_DIR / "checkpoints" / f"{EXPERIMENT_NAME}.pt"
LAST_PATH: Path = LAB_DIR / "checkpoints" / f"{EXPERIMENT_NAME}-last.pt"  # resumable training state, rewritten every epoch

N_CLASSES: int = 20
IGNORE_INDEX: int = 255
CLASS_NAMES: list[str] = [
    "Background", "Hat", "Hair", "Glove", "Sunglasses", "Upper-clothes", "Dress", "Coat", "Socks", "Pants",
    "Torso-skin", "Scarf", "Skirt", "Face", "Left-arm", "Right-arm", "Left-leg", "Right-leg", "Left-shoe", "Right-shoe",
]
MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
PAD_COLOR: tuple[int, int, int] = (124, 116, 104)  # grey border already present in the prepared images
FLIP_LUT: np.ndarray = np.arange(256, dtype=np.uint8)
FLIP_LUT[[14, 15, 16, 17, 18, 19]] = [15, 14, 17, 16, 19, 18]
FLIP_CHANNEL_PERM: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 17, 16, 19, 18]


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
    torch.backends.cudnn.benchmark = True


# ============================================================================ #
#                                   Data                                       #
# ============================================================================ #
def list_ids(folder: Path) -> list[str]:
    ids = sorted(p.stem for p in folder.glob("*.png"))
    if not ids:
        raise FileNotFoundError(f"no PNG images in {folder}")
    return ids


class CIHPSegDataset(Dataset):
    """Image + mask pairs; label-aware horizontal flip, then albumentations, then normalise."""

    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        ids: list[str],
        transform: A.Compose | None,
        hflip_prob: float,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.ids = ids
        self.transform = transform
        self.hflip_prob = hflip_prob
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.ids[index]
        with Image.open(self.image_dir / f"{stem}.png") as handle:
            image = np.array(handle.convert("RGB"))
        with Image.open(self.mask_dir / f"{stem}.png") as handle:
            mask = np.array(handle, dtype=np.uint8)
        if random.random() < self.hflip_prob:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(FLIP_LUT[mask[:, ::-1]])
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
        tensor = (tensor - self.mean) / self.std
        return tensor, torch.from_numpy(np.ascontiguousarray(mask)).long()


class CIHPTestDataset(Dataset):
    def __init__(self, image_dir: Path, ids: list[str]) -> None:
        self.image_dir = image_dir
        self.ids = ids
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.image_dir / f"{self.ids[index]}.png") as handle:
            image = np.array(handle.convert("RGB"))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        return (tensor - self.mean) / self.std


def build_train_transform(cfg: ExperimentConfig) -> A.Compose:
    return A.Compose([
        A.Affine(
            scale=cfg.scale_range,
            translate_percent=(-cfg.translate, cfg.translate),
            rotate=(-cfg.rotate_deg, cfg.rotate_deg),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=PAD_COLOR,
            fill_mask=IGNORE_INDEX,
            p=cfg.affine_p,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=25,
            val_shift_limit=10,
            p=0.5,
        ),
    ])


# ============================================================================ #
#                                   Model                                      #
# ============================================================================ #
class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of a (B, C, H, W) tensor, as in SAM."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=dim, out_features=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden, out_features=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    """Multi-head self-attention inside non-overlapping windows; window 0 means global."""

    def __init__(self, dim: int, heads: int, window: int) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.window = window
        self.qkv = nn.Linear(in_features=dim, out_features=3 * dim)
        self.proj = nn.Linear(in_features=dim, out_features=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, H, W, C)
        b, h, w, c = x.shape
        win = self.window if self.window > 0 else h
        assert h % win == 0 and w % win == 0, f"window {win} must divide {h}x{w}"
        x = x.view(b, h // win, win, w // win, win, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, win * win, c)
        qkv = self.qkv(x).reshape(x.shape[0], win * win, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        out = out.transpose(1, 2).reshape(x.shape[0], win * win, c)
        out = self.proj(out)
        return out.view(b, h // win, w // win, win, win, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)


class AttnBlock(nn.Module):
    """Depthwise-conv positional encoding + windowed/global attention + MLP, pre-norm with LayerScale."""

    def __init__(
        self,
        dim: int,
        heads: int,
        window: int,
        mlp_ratio: float,
        layer_scale_init: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.cpe = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim=dim, heads=heads, window=window)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, hidden=int(dim * mlp_ratio))
        self.ls1 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.ls2 = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(p=drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, H, W, C)
        x = x + self.cpe(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.ls1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ls2 * self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):
    """ConvNeXt block (7x7 depthwise conv, LayerNorm, MLP), the convolutional fallback for early stages."""

    def __init__(self, dim: int, mlp_ratio: float, layer_scale_init: float, drop_path: float) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, hidden=int(dim * mlp_ratio))
        self.ls = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(p=drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, H, W, C)
        y = self.dwconv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        y = self.mlp(self.norm(y))
        return x + self.drop_path(self.ls * y)


class Downsample(nn.Module):
    """Hiera-style stage transition: 2x2 max-pool of the tokens, then a linear projection."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_features=dim_in, out_features=dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, H, W, C)
        x = F.max_pool2d(x.permute(0, 3, 1, 2), kernel_size=2).permute(0, 2, 3, 1)
        return self.proj(x)


class ConvStem(nn.Module):
    """Three 3x3 convolutions to stride 4 ('early convolutions help transformers see better')."""

    def __init__(self, stem_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=stem_dim),
            nn.GELU(),
            nn.Conv2d(in_channels=stem_dim, out_channels=stem_dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=stem_dim * 2),
            nn.GELU(),
            nn.Conv2d(in_channels=stem_dim * 2, out_channels=out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 3, H, W) -> (B, H/4, W/4, C)
        return self.net(x).permute(0, 2, 3, 1)


class HierarchicalEncoder(nn.Module):
    """Four stages at strides 4/8/16/32; returns channels-first feature maps of every stage."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.stem = ConvStem(stem_dim=cfg.stem_dim, out_dim=cfg.dims[0])
        total_blocks = sum(cfg.depths)
        drop_rates = [cfg.drop_path_rate * i / max(total_blocks - 1, 1) for i in range(total_blocks)]
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        block_index = 0
        for stage, (dim, depth, heads, window, kind) in enumerate(
            zip(cfg.dims, cfg.depths, cfg.heads, cfg.windows, cfg.stage_types)
        ):
            blocks: list[nn.Module] = []
            for _ in range(depth):
                if kind == "attn":
                    blocks.append(AttnBlock(
                        dim=dim,
                        heads=heads,
                        window=window,
                        mlp_ratio=cfg.mlp_ratio,
                        layer_scale_init=cfg.layer_scale_init,
                        drop_path=drop_rates[block_index],
                    ))
                elif kind == "conv":
                    blocks.append(ConvBlock(
                        dim=dim,
                        mlp_ratio=cfg.mlp_ratio,
                        layer_scale_init=cfg.layer_scale_init,
                        drop_path=drop_rates[block_index],
                    ))
                else:
                    raise ValueError(kind)
                block_index += 1
            self.stages.append(nn.Sequential(*blocks))
            if stage + 1 < len(cfg.dims):
                self.downsamples.append(Downsample(dim_in=dim, dim_out=cfg.dims[stage + 1]))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for dim in cfg.dims)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features: list[torch.Tensor] = []
        for stage, blocks in enumerate(self.stages):
            x = blocks(x)
            features.append(self.norms[stage](x).permute(0, 3, 1, 2).contiguous())
            if stage < len(self.downsamples):
                x = self.downsamples[stage](x)
        return features


class FPNNeck(nn.Module):
    """SAM 2-style neck: 1x1 laterals to a common width, top-down nearest-upsample adds, 3x3 smoothing."""

    def __init__(self, in_dims: tuple[int, ...], dim: int) -> None:
        super().__init__()
        self.laterals = nn.ModuleList(
            nn.Sequential(nn.Conv2d(in_channels=d, out_channels=dim, kernel_size=1, bias=False), nn.BatchNorm2d(num_features=dim))
            for d in in_dims
        )
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_features=dim),
                nn.GELU(),
            )
            for _ in in_dims
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = [lateral(f) for lateral, f in zip(self.laterals, features)]
        for level in range(len(laterals) - 1, 0, -1):
            laterals[level - 1] = laterals[level - 1] + F.interpolate(laterals[level], scale_factor=2.0, mode="nearest")
        return [smooth(l) for smooth, l in zip(self.smooth, laterals)]


class CrossAttention(nn.Module):
    """Attention with an optional internal down-projection, as in SAM's two-way transformer."""

    def __init__(self, dim: int, heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        inner = dim // downsample_rate
        assert inner % heads == 0
        self.heads = heads
        self.q_proj = nn.Linear(in_features=dim, out_features=inner)
        self.k_proj = nn.Linear(in_features=dim, out_features=inner)
        self.v_proj = nn.Linear(in_features=dim, out_features=inner)
        self.out_proj = nn.Linear(in_features=inner, out_features=dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, nq, _ = q.shape
        nk = k.shape[1]
        q = self.q_proj(q).view(b, nq, self.heads, -1).transpose(1, 2)
        k = self.k_proj(k).view(b, nk, self.heads, -1).transpose(1, 2)
        v = self.v_proj(v).view(b, nk, self.heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, nq, -1)
        return self.out_proj(out)


class TwoWayBlock(nn.Module):
    """SAM decoder block: token self-attn, token->image cross-attn, token MLP, image->token cross-attn."""

    def __init__(self, dim: int, heads: int, mlp_dim: int) -> None:
        super().__init__()
        self.self_attn = CrossAttention(dim=dim, heads=heads)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_t2i = CrossAttention(dim=dim, heads=heads, downsample_rate=2)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim=dim, hidden=mlp_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.cross_i2t = CrossAttention(dim=dim, heads=heads, downsample_rate=2)
        self.norm4 = nn.LayerNorm(dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = queries + query_pe
        queries = self.norm1(queries + self.self_attn(q=q, k=q, v=queries))
        q = queries + query_pe
        k = keys + key_pe
        queries = self.norm2(queries + self.cross_t2i(q=q, k=k, v=keys))
        queries = self.norm3(queries + self.mlp(queries))
        q = queries + query_pe
        k = keys + key_pe
        keys = self.norm4(keys + self.cross_i2t(q=k, k=q, v=queries))
        return queries, keys


class MaskDecoder(nn.Module):
    """Two-way transformer over 20 class tokens and the stride-16 embedding; masks = tokens . pixel embedding."""

    def __init__(self, cfg: ExperimentConfig, grid: int) -> None:
        super().__init__()
        dim = cfg.neck_dim
        self.class_tokens = nn.Parameter(torch.zeros(1, N_CLASSES, dim))
        self.image_pe = nn.Parameter(torch.zeros(1, grid * grid, dim))
        nn.init.trunc_normal_(self.class_tokens, std=0.02)
        nn.init.trunc_normal_(self.image_pe, std=0.02)
        self.blocks = nn.ModuleList(
            TwoWayBlock(dim=dim, heads=cfg.decoder_heads, mlp_dim=cfg.decoder_mlp_dim) for _ in range(cfg.decoder_depth)
        )
        self.final_attn = CrossAttention(dim=dim, heads=cfg.decoder_heads, downsample_rate=2)
        self.final_norm = nn.LayerNorm(dim)
        # upscaling path stride 16 -> 8 -> 4, fed with the FPN high-resolution features as in SAM 2
        self.up1 = nn.ConvTranspose2d(in_channels=dim, out_channels=dim // 2, kernel_size=2, stride=2)
        self.skip8 = nn.Conv2d(in_channels=dim, out_channels=dim // 2, kernel_size=1)
        self.norm_up1 = LayerNorm2d(dim // 2)
        self.up2 = nn.ConvTranspose2d(in_channels=dim // 2, out_channels=cfg.mask_dim, kernel_size=2, stride=2)
        self.skip4 = nn.Conv2d(in_channels=dim, out_channels=cfg.mask_dim, kernel_size=1)
        self.act = nn.GELU()
        self.hypernet = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.Linear(in_features=dim, out_features=cfg.mask_dim),
        )

    def forward(self, feat4: torch.Tensor, feat8: torch.Tensor, feat16: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat16.shape
        keys = feat16.flatten(2).transpose(1, 2)  # (B, HW, C)
        queries = self.class_tokens.expand(b, -1, -1)
        query_pe = torch.zeros_like(queries)
        key_pe = self.image_pe.expand(b, -1, -1)
        for block in self.blocks:
            queries, keys = block(queries=queries, keys=keys, query_pe=query_pe, key_pe=key_pe)
        q = queries + query_pe
        k = keys + key_pe
        queries = self.final_norm(queries + self.final_attn(q=q, k=k, v=keys))
        image = keys.transpose(1, 2).reshape(b, c, h, w)
        image = self.act(self.norm_up1(self.up1(image) + self.skip8(feat8)))
        pixel = self.act(self.up2(image) + self.skip4(feat4))  # (B, mask_dim, H/4, W/4)
        mask_embed = self.hypernet(queries)  # (B, 20, mask_dim)
        return torch.einsum("bqc,bchw->bqhw", mask_embed, pixel)


class MiniSAM(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.encoder = HierarchicalEncoder(cfg=cfg)
        self.neck = FPNNeck(in_dims=cfg.dims, dim=cfg.neck_dim)
        self.decoder = MaskDecoder(cfg=cfg, grid=cfg.img_size // 16)
        self.aux_head = nn.Conv2d(in_channels=cfg.neck_dim, out_channels=N_CLASSES, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        size = x.shape[-2:]
        feats = self.neck(self.encoder(x))
        logits = self.decoder(feat4=feats[0], feat8=feats[1], feat16=feats[2])
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        if not self.training:
            return logits, None
        aux = F.interpolate(self.aux_head(feats[0]), size=size, mode="bilinear", align_corners=False)
        return logits, aux


# ============================================================================ #
#                               Loss / metrics                                 #
# ============================================================================ #
def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=1)
    valid = (targets != IGNORE_INDEX).unsqueeze(1).float()
    safe = torch.where(targets == IGNORE_INDEX, torch.zeros_like(targets), targets)
    one_hot = F.one_hot(safe, num_classes=N_CLASSES).permute(0, 3, 1, 2).float()
    probs = probs * valid
    one_hot = one_hot * valid
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    cardinality = (probs + one_hot).sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + eps) / (cardinality + eps)).mean()


def lovasz_softmax_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-image Lovasz-softmax over the classes present in each image, batched over images and classes.

    Invalid (255) pixels get error -1 so they sort last and contribute nothing; the result is the mean over images
    of the mean over present classes of the Lovasz extension of that class's Jaccard loss.
    """
    probs = F.softmax(logits.float(), dim=1).flatten(2)  # (B, C, P)
    valid = (targets != IGNORE_INDEX).flatten(1).float().unsqueeze(1)  # (B, 1, P)
    safe = torch.where(targets == IGNORE_INDEX, torch.zeros_like(targets), targets).flatten(1)
    fg = F.one_hot(safe, num_classes=N_CLASSES).permute(0, 2, 1).float() * valid  # (B, C, P)
    errors = (fg - probs).abs() * valid + (valid - 1.0)
    errors_sorted, perm = errors.sort(dim=-1, descending=True)
    gt_sorted = fg.gather(dim=-1, index=perm)
    gts = gt_sorted.sum(dim=-1, keepdim=True)
    intersection = gts - gt_sorted.cumsum(dim=-1)
    union = gts + (1.0 - gt_sorted).cumsum(dim=-1)
    jaccard = 1.0 - intersection / union
    grad = torch.cat([jaccard[..., :1], jaccard[..., 1:] - jaccard[..., :-1]], dim=-1)
    per_class = (errors_sorted.clamp(min=0.0) * grad).sum(dim=-1)  # (B, C)
    present = (gts.squeeze(-1) > 0).float()
    per_image = (per_class * present).sum(dim=1) / present.sum(dim=1).clamp(min=1.0)
    return per_image.mean()


def seg_loss(logits: torch.Tensor, targets: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    ce = F.cross_entropy(logits.float(), targets, ignore_index=IGNORE_INDEX)
    return cfg.ce_weight * ce + cfg.lovasz_weight * lovasz_softmax_loss(logits=logits, targets=targets)


def confusion_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(B, H, W) predictions and targets -> (B, 20, 20) confusion matrices, ignoring 255."""
    valid = target != IGNORE_INDEX
    index = target.clamp(max=N_CLASSES - 1) * N_CLASSES + pred
    index = torch.where(valid, index, torch.full_like(index, N_CLASSES * N_CLASSES))
    counts = torch.stack([torch.bincount(row.flatten(), minlength=N_CLASSES * N_CLASSES + 1) for row in index])
    return counts[:, : N_CLASSES * N_CLASSES].view(-1, N_CLASSES, N_CLASSES)


def miou_from_confusion(conf: torch.Tensor) -> torch.Tensor:
    """Per-image mIoU (course protocol) from (B, 20, 20) confusion matrices."""
    inter = conf.diagonal(dim1=1, dim2=2).float()
    union = conf.sum(dim=1).float() + conf.sum(dim=2).float() - inter
    present = union > 0
    iou = torch.where(present, inter / union.clamp(min=1), torch.zeros_like(inter))
    return iou.sum(dim=1) / present.sum(dim=1).clamp(min=1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: ExperimentConfig,
    tta_hflip: bool = False,
) -> tuple[float, float, float, torch.Tensor]:
    """Returns (mean per-image mIoU, pixel accuracy, mean loss, global per-class IoU)."""
    model.eval()
    mious: list[torch.Tensor] = []
    total_conf = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long, device=device)
    loss_sum = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = predict_logits(model=model, images=images, tta_hflip=tta_hflip)
        loss_sum += seg_loss(logits=logits, targets=masks, cfg=cfg).item() * images.shape[0]
        conf = confusion_per_image(pred=logits.argmax(dim=1), target=masks)
        mious.append(miou_from_confusion(conf=conf))
        total_conf += conf.sum(dim=0)
    miou_all = torch.cat(mious)
    inter = total_conf.diagonal().float()
    union = total_conf.sum(dim=0).float() + total_conf.sum(dim=1).float() - inter
    per_class = inter / union.clamp(min=1)
    pixel_acc = (inter.sum() / total_conf.sum().clamp(min=1)).item()
    return miou_all.mean().item(), pixel_acc, loss_sum / len(loader.dataset), per_class.cpu()


def predict_logits(model: nn.Module, images: torch.Tensor, tta_hflip: bool) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(images)
        logits = logits.float()
        if tta_hflip:
            flipped, _ = model(images.flip(dims=[3]))
            flipped = flipped.float().flip(dims=[3])[:, FLIP_CHANNEL_PERM]
            logits = 0.5 * (logits + flipped)
    return logits


# ============================================================================ #
#                               Optimisation                                   #
# ============================================================================ #
def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".ls") or ".ls1" in name or ".ls2" in name or "tokens" in name or "_pe" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step: int, total_steps: int, warmup_steps: int, cfg: ExperimentConfig) -> float:
    if step < warmup_steps:
        return cfg.lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))


class EMA:
    """Exponential moving average of the parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


def save_training_state(
    path: Path,
    model: nn.Module,
    ema: EMA,
    optimizer: optim.Optimizer,
    epoch: int,
    step: int,
    best_miou: float,
    best_tag: str,
    best_state: dict[str, torch.Tensor] | None,
) -> None:
    """Atomically write everything needed to continue training after the given epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_miou": best_miou,
        "best_tag": best_tag,
        "best_state": best_state,
    }, tmp)
    os.replace(tmp, path)


def format_per_class(per_class: torch.Tensor) -> str:
    return " ".join(f"{name[:6]}={iou:.2f}" for name, iou in zip(CLASS_NAMES, per_class.tolist()))


# ============================================================================ #
#                                   Main                                       #
# ============================================================================ #
def main(cfg: ExperimentConfig, smoke: bool) -> None:
    start = time.time()
    set_seed(cfg.seed)
    device = torch.device("cuda")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Config: {asdict(cfg)}")
    print(f"Device: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

    # ---------------------------------------------------------------- data
    train_ids = list_ids(folder=DATA_DIR / "train")
    val_ids = list_ids(folder=DATA_DIR / "val")
    if smoke:
        train_ids, val_ids = train_ids[:512], val_ids[:64]
    train_set = CIHPSegDataset(
        image_dir=DATA_DIR / "train",
        mask_dir=DATA_DIR / "train_mask",
        ids=train_ids,
        transform=build_train_transform(cfg=cfg),
        hflip_prob=0.5,
    )
    val_set = CIHPSegDataset(
        image_dir=DATA_DIR / "val",
        mask_dir=DATA_DIR / "val_mask",
        ids=val_ids,
        transform=None,
        hflip_prob=0.0,
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    print(f"train {len(train_set)} / val {len(val_set)} images, {len(train_loader)} steps per epoch")

    # --------------------------------------------------------------- model
    model = MiniSAM(cfg=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MiniSAM params: {n_params / 1e6:.2f}M")
    optimizer = optim.AdamW(params=param_groups(model=model, weight_decay=cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.999))
    ema = EMA(model=model, decay=cfg.ema_decay)
    epochs = 1 if smoke else cfg.epochs
    total_steps = epochs * len(train_loader)
    warmup_steps = cfg.warmup_epochs * len(train_loader)

    best_miou = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    best_tag = ""
    step = 0
    start_epoch = 0
    if not smoke and LAST_PATH.is_file():
        state = torch.load(LAST_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
        optimizer.load_state_dict(state["optimizer"])
        best_miou, best_tag, best_state = state["best_miou"], state["best_tag"], state["best_state"]
        step, start_epoch = state["step"], state["epoch"] + 1
        print(f"Resumed from {LAST_PATH.name}: {state['epoch'] + 1} epochs done, best val mIoU {best_miou:.4f} ({best_tag})")
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_start = time.time()
        loss_sum = 0.0
        seen = 0
        for images, masks in train_loader:
            lr = lr_at(step=step, total_steps=total_steps, warmup_steps=warmup_steps, cfg=cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, aux = model(images)
            loss = seg_loss(logits=logits, targets=masks, cfg=cfg)
            if aux is not None and cfg.aux_weight > 0:
                loss = loss + cfg.aux_weight * F.cross_entropy(aux.float(), masks, ignore_index=IGNORE_INDEX)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            ema.update(model=model)
            loss_sum += loss.item() * images.shape[0]
            seen += images.shape[0]
            step += 1
            if smoke and step == 30:
                torch.cuda.synchronize()
                print(f"smoke: 30 steps in {time.time() - epoch_start:.1f}s -> "
                      f"{30 * cfg.batch_size / (time.time() - epoch_start):.0f} img/s, "
                      f"peak mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, loss {loss.item():.3f}")
                break
        train_loss = loss_sum / max(seen, 1)
        train_time = time.time() - epoch_start

        raw_miou, raw_acc, raw_loss, raw_per_class = evaluate(model=model, loader=val_loader, device=device, cfg=cfg)
        raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
        ema_miou, ema_acc, ema_loss, ema_per_class = evaluate(model=model, loader=val_loader, device=device, cfg=cfg)
        model.load_state_dict(raw_state)

        if ema_miou >= raw_miou:
            epoch_best, tag, per_class = ema_miou, "ema", ema_per_class
        else:
            epoch_best, tag, per_class = raw_miou, "raw", raw_per_class
        record = epoch_best > best_miou
        if record:
            best_miou = epoch_best
            best_tag = tag
            best_state = ema.state_dict() if tag == "ema" else raw_state
        print(
            f"epoch {epoch + 1}/{epochs} | lr {lr:.2e} | train loss {train_loss:.3f} | "
            f"val raw {raw_miou:.4f}/{raw_acc:.4f}/{raw_loss:.3f} ema {ema_miou:.4f}/{ema_acc:.4f}/{ema_loss:.3f} "
            f"(mIoU/acc/loss) | {train_time / 60:.1f} min{' *' if record else ''}"
        )
        if record or (epoch + 1) % 5 == 0:
            print(f"  per-class IoU ({tag}): {format_per_class(per_class=per_class)}")
        if not smoke:
            save_training_state(
                path=LAST_PATH,
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                best_miou=best_miou,
                best_tag=best_tag,
                best_state=best_state,
            )

    if smoke:
        print(f"smoke done in {(time.time() - start) / 60:.1f} min")
        return

    assert best_state is not None
    print(f"Best val mIoU: {best_miou:.4f} ({best_tag})")

    # ------------------------------------------------------------ test-time
    model.load_state_dict(best_state)
    tta_miou, tta_acc, _, _ = evaluate(model=model, loader=val_loader, device=device, cfg=cfg, tta_hflip=cfg.tta_hflip)
    print(f"Val mIoU with hflip TTA: {tta_miou:.4f} (acc {tta_acc:.4f}); without: {best_miou:.4f}")
    use_tta = cfg.tta_hflip and tta_miou >= best_miou
    final_val = max(tta_miou, best_miou) if cfg.tta_hflip else best_miou

    # --------------------------------------------------------------- output
    is_record = final_val > CURRENT_BEST_VAL
    print(f"Current best: {CURRENT_BEST_VAL:.4f} -> new record: {is_record}")
    if WRITE_OUTPUTS:  # every run keeps its CSV and checkpoint (ensembling later); is_record is informational
        assert not CSV_PATH.exists() and not CKPT_PATH.exists(), "refuse to overwrite"
        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(cfg), "state_dict": best_state, "val_miou": final_val, "tta": use_tta}, CKPT_PATH)

        test_index = list(csv.DictReader((DATA_DIR / "test_index.csv").open(newline="")))
        test_ids = [row["filename"].removesuffix(".png") for row in test_index]
        assert [int(row["index"]) for row in test_index] == list(range(len(test_ids)))
        test_loader = DataLoader(
            dataset=CIHPTestDataset(image_dir=DATA_DIR / "test", ids=test_ids),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
        )
        model.eval()
        row_index = 0
        with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "pred_mask"])
            with torch.no_grad():
                for images in test_loader:
                    logits = predict_logits(model=model, images=images.to(device, non_blocking=True), tta_hflip=use_tta)
                    for pred in logits.argmax(dim=1).cpu().numpy().astype(np.uint8):
                        writer.writerow([row_index, json.dumps(pred.tolist(), separators=(",", ":"))])
                        row_index += 1
        assert row_index == len(test_ids)
        print(f"Saved {CSV_PATH.name} ({row_index} rows) and {CKPT_PATH.name}")

    print(f"RESULT {EXPERIMENT_NAME} val={final_val:.4f} best_epoch_val={best_miou:.4f} tta={use_tta}")
    print(f"Elapsed: {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="30 training steps + tiny validation, no outputs")
    args = parser.parse_args()
    if args.smoke:
        main(CONFIG, smoke=True)
    else:
        resuming = LAST_PATH.is_file()
        assert resuming or not LOG_PATH.exists(), f"{LOG_PATH} already exists, refuse to overwrite"
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a" if resuming else "w") as log_file:
            sys.stdout = Tee(sys.stdout, log_file)
            sys.stderr = Tee(sys.stderr, log_file)
            main(CONFIG, smoke=False)
