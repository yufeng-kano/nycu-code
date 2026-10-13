"""Compute the validation confusion matrix of the best experiment checkpoint (CPU only)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

LAB_DIR: Path = Path(__file__).resolve().parent.parent
EXPERIMENT: str = "016-erasing-0.5"

spec = importlib.util.spec_from_file_location(name="exp", location=LAB_DIR / "scripts" / f"{EXPERIMENT}.py")
exp = importlib.util.module_from_spec(spec)
sys.modules["exp"] = exp
spec.loader.exec_module(exp)
cfg = exp.CONFIG

stats_transform = transforms.Compose([
    transforms.Resize(size=(cfg.img_size, cfg.img_size)),
    transforms.ToImage(),
    transforms.ToDtype(dtype=torch.float32, scale=True),
])
mean, std = exp.compute_mean_std(
    dataset=exp.IntelData(root=exp.DATA_DIR, split="train", transform=stats_transform),
)
_, transforms_test = exp.build_transforms(cfg=cfg, mean=mean, std=std)
valid_set = exp.IntelData(root=exp.DATA_DIR, split="val", mode="train", transform=transforms_test)
val_loader = DataLoader(dataset=valid_set, batch_size=100, shuffle=False, num_workers=4)

model = exp.YourCNNModel(cfg=cfg)
model.load_state_dict(torch.load(LAB_DIR / "checkpoints" / f"{EXPERIMENT}.pt", map_location="cpu", weights_only=True))
model.eval()

num_classes: int = len(exp.CLASS_NAMES)
confusion: np.ndarray = np.zeros(shape=(num_classes, num_classes), dtype=np.int64)
with torch.no_grad():
    for images, labels in val_loader:
        predicted: torch.Tensor = model(images).argmax(dim=1)
        np.add.at(confusion, (labels.numpy(), predicted.numpy()), 1)

print(f"val acc: {np.trace(confusion) / confusion.sum():.4f}")
print(confusion)
np.savetxt(
    fname=Path(__file__).resolve().parent / "figures" / "confusion_val.csv",
    X=confusion,
    fmt="%d",
    delimiter=",",
)
