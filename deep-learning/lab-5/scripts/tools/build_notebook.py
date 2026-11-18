"""Build notebook/HW5_315834011.ipynb from the TA template.

The template file is never modified and its cells stay in place (the YOLOv3 baseline with the filled TODOs
runs as shipped), as in lab-4: only the name cell is filled, and the final detector (the anchor-free main line
in `notebook/src/det_*.py`, the recipe of run 016 on a 30-epoch schedule, trained on train + val) is appended
after the template's Kaggle cell; it runs on the local RTX 3090 Ti (24 GB) from the notebook folder. Cells are addressed by index after verifying a distinctive
substring, so a template change fails loudly.

Usage: build_notebook.py [OUT.ipynb] [--force]
An existing output is never overwritten unless --force is passed, because the executed notebook is
the training record the TA grades.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = LAB / "notebook"
TEMPLATE = NOTEBOOK_DIR / "Lab5_object_detection_coco.ipynb"
TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else NOTEBOOK_DIR / "HW5_315834011.ipynb"


def set_source(cell: dict, src: str) -> None:
    cell["source"] = src.rstrip("\n").splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def markdown(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.rstrip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.rstrip("\n").splitlines(keepends=True),
    }


APPROACH_MD = """\
## Final Detector

The YOLOv3 cells above are the template with its TODO blocks filled in `src/yolo.py` and `src/yolo_loss.py` (run 001 of the report: validation AP 0.120). The cells below train the detector that replaced it, implemented from scratch in `src/`:

- `src/detector.py`: ImageNet-1k ConvNeXt backbone from timm (`*.fb_in1k`, supervised classification weights only, strides 8 / 16 / 32) -> PAFPN neck (C2f-style CSP blocks, 384 channels, depth 3) -> decoupled head that predicts class logits and a discrete distribution over the four box distances (DFL, 16 bins) per cell; anchor-free, multi-label scoring, class-aware NMS.
- `src/det_loss.py`: task-aligned assignment (top-10 cells per box by `score^0.5 * IoU^6`), BCE classification against the alignment scores, CIoU box loss and distribution focal loss (weights 0.5 / 7.5 / 1.5).
- `src/det_data.py`: letterbox, mosaic of four images on a 2x canvas, box-level copy-paste (up to 6 object crops from other images pasted on every mosaic), random affine, mixup between two mosaics, HSV jitter, horizontal flip; mosaic is switched off for the last 10 epochs.
- `src/det_train.py`: the training loop (`TrainConfig` holds every setting): batch-level multi-scale 0.6-1.25 of the letterbox size, AdamW with a 0.25x backbone learning rate, 3 warmup + cosine epochs, bf16 autocast, gradient checkpointing, EMA weights evaluated every 5 epochs.
- `src/det_eval.py`: COCO AP on original-image coordinates and the Kaggle CSV, with test-time augmentation (horizontal flip and two extra scales of the same model, pooled by one NMS). One model, no ensemble.

The recipe was chosen by one-change-at-a-time experiments on an RTX 3090 Ti (`scripts/train/001-017`, table in the report). Run 016 (ConvNeXt-B, 800 letterbox, batch 12, copy-paste on every mosaic, 40 epochs) reached validation AP 0.440, 0.458 with TTA at 896; run 017 showed that a 30-epoch cosine ends at its peak (last epoch 0.441, equal to the best), so the schedule can be fixed in advance. This notebook is one end-to-end run of that recipe on the 3090 Ti (ConvNeXt-B, 800 letterbox with multi-scale 480-1000, batch 12, 30 epochs) trained on train + val minus a seeded hold-out of 400 val images. Because the full validation set is no longer available for model selection, the schedule is fixed and the EMA weights of the last epoch are the model; the hold-out AP printed every 5 epochs is a (noisy) monitor only.
"""

IMPORTS = """\
import json
from dataclasses import asdict, replace

from src.config import NUM_CLASSES
from src.det_eval import evaluate_detector, write_submission
from src.det_loss import DetectionLoss
from src.det_train import (
    TrainConfig,
    build_datasets,
    build_model,
    build_optimizer,
    make_loader,
    subset_coco,
    train_detector,
)
"""

CONFIG_MD = """\
### Configuration

`TrainConfig()` defaults are the 3090 Ti recipe of run 015; the overrides below add run 016's copy-paste dose, the 30-epoch schedule of run 017 and the train + val split. Peak memory with gradient checkpointing is about 21 GB at batch 12 (inputs up to 1000), so it fills the 24 GB card; with 13,190 training images an epoch takes about 12 min, the whole run about 6.5 h. Training resumes from `checkpoints/final-last.pt` if the kernel is restarted.
"""

CONFIG = """\
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_root = Path("dataset")

# train_on_val: train on train.txt + val.txt minus a 400-image hold-out (seeded split, written to the run folder).
# select="last": the EMA weights of the last epoch are the model (no full validation set to pick an epoch).
cfg = TrainConfig(
    copy_paste=1.0,  # run 016: every mosaic, up to 6 crops (best 896 TTA, flatter plateau than 0.5)
    copy_paste_objects=6,
    epochs=30,  # run 017: the 30-epoch cosine ends at its peak
    train_on_val=True,
    select="last",
)

# Final evaluation and submission: 896 letterbox, horizontal flip and scales 0.75 / 1.1, pooled by one class-aware NMS
# (models trained with multi-scale up to 1000 scored best at 896 with this TTA, runs 008-017).
eval_image_size = 896
tta_scales = (0.75, 1.1)

# True runs the whole pipeline on a few images for 2 short epochs (pipeline check only).
smoke = False
if smoke:
    cfg = replace(cfg, batch_size=2, num_workers=2)

det_checkpoint_dir = Path("checkpoints")  # a run resumes from <run_name>-last.pt after a kernel restart
run_name = "final"
log_path = det_checkpoint_dir / f"{run_name}.log"  # the printed training lines are also appended here
RESULT_PATH = Path("result.csv")

print(json.dumps(asdict(cfg), indent=2))
"""

DATASET_MD = """\
### Dataset & DataLoader

`DetectionDataset` returns the letterboxed image, an `(n, 5)` target `[class, x1, y1, x2, y2]` in input pixels and the metadata needed to map boxes back to the original image. With `cfg.train_on_val` the training list is train.txt plus val.txt minus the hold-out (`trainval.txt` / `holdout.txt` in the run folder); the hold-out is the validation set of the cells below.

`train_detector` builds its own training and validation loaders from `cfg`; the loaders here are for the final evaluation and the Kaggle submission at `eval_image_size`.
"""

DATASET = """\
det_checkpoint_dir.mkdir(parents=True, exist_ok=True)
train_dataset, val_dataset, test_dataset = build_datasets(cfg, data_root, split_dir=det_checkpoint_dir)

eval_cfg = replace(cfg, image_size=eval_image_size)
_, val_eval_dataset, test_eval_dataset = build_datasets(eval_cfg, data_root, split_dir=det_checkpoint_dir)
limit = 32 if smoke else None
val_loader = make_loader(val_eval_dataset, eval_cfg, shuffle=False, limit=limit)
test_loader = make_loader(test_eval_dataset, eval_cfg, shuffle=False, limit=limit)
val_annotations = subset_coco(val_eval_dataset, limit) if smoke else val_eval_dataset.coco

print(f"Train (train + val minus hold-out): {len(train_dataset):,} images")
print(f"Validation (hold-out): {len(val_dataset):,} images")
print(f"Test: {len(test_dataset):,} images")
print(f"Classes: {NUM_CLASSES}")
"""

INSPECT_MD = """\
### Inspect an Annotated Image

Left: a hold-out image letterboxed to the training size with its ground-truth boxes. Right: a training sample after mosaic, copy-paste, random affine, HSV jitter and flip (and mixup with a second mosaic when drawn).
"""

INSPECT = """\
def to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.permute(1, 2, 0).numpy()
    return (image * np.array(IMG_STD) + np.array(IMG_MEAN)).clip(0, 1)


def draw_target(axis, image: np.ndarray, target: torch.Tensor, title: str) -> None:
    colors = plt.get_cmap("tab20")
    axis.imshow(image)
    for label, x1, y1, x2, y2 in target.tolist():
        color = colors(int(label))
        axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.5))
        axis.text(
            x1,
            y1,
            CLASS_NAMES[int(label)],
            fontsize=8,
            color="#111111",
            va="top",
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.9, "pad": 1},
            clip_on=True,
        )
    axis.set_title(title)
    axis.axis("off")


random.seed(cfg.seed)
np.random.seed(cfg.seed)
val_image, val_target, val_metadata = val_dataset[0]
train_image, train_target, _ = train_dataset[0]

figure, axes = plt.subplots(1, 2, figsize=(16, 8))
draw_target(axes[0], to_rgb(val_image), val_target, f"hold-out, letterbox {cfg.image_size}: {val_metadata['file_name']}")
draw_target(axes[1], to_rgb(train_image), train_target, "training sample: mosaic + copy-paste + affine + HSV + flip")
plt.tight_layout()
plt.show()
"""

MODEL_MD = """\
### Model

`Detector` = timm backbone (strides 8 / 16 / 32, ImageNet-1k weights) -> PAFPN -> decoupled head. The forward pass returns, flattened over the three levels, class logits `(N, A, 20)` and DFL logits `(N, A, 4 * 16)` together with the cell centres and strides; `inference` decodes them to boxes and runs class-aware NMS.
"""

MODEL = """\
model = build_model(cfg, pretrained=True).to(device)

with torch.no_grad():
    outputs = model(torch.zeros(1, 3, cfg.image_size, cfg.image_size, device=device))

print(f"Device: {device}")
print(f"Backbone: {cfg.model_name}, feature channels {model.backbone.channels}")
print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
print(f"Outputs at {cfg.image_size}: cls {tuple(outputs['cls'].shape)}, reg {tuple(outputs['reg'].shape)}")
"""

LOSS_MD = """\
### Loss, Optimizer & Scheduler

Targets are assigned by the task-aligned assigner inside `DetectionLoss`; the loss is BCE against the alignment scores plus CIoU and DFL over the positive cells, normalised by the sum of target scores.

AdamW with a 0.25x learning rate on the backbone and no weight decay on norms and biases; 3 warmup epochs, then cosine to `min_lr`, stepped every iteration. The EMA of the weights is what gets evaluated and saved.
"""

LOSS = """\
criterion = DetectionLoss(
    num_classes=NUM_CLASSES,
    reg_max=cfg.reg_max,
    box_weight=cfg.box_weight,
    cls_weight=cfg.cls_weight,
    dfl_weight=cfg.dfl_weight,
    top_k=cfg.top_k,
    alpha=cfg.alpha,
    beta=cfg.beta,
)
optimizer = build_optimizer(model, cfg)
for group in optimizer.param_groups:
    count = sum(parameter.numel() for parameter in group["params"])
    print(f"lr {group['lr']:.1e} | weight decay {group['weight_decay']:.2f} | {count:,} parameters")

# train_detector rebuilds the model, loss, optimizer and EMA from cfg (so a run can resume from <run_name>-last.pt).
del model, optimizer
"""

TRAIN_MD = """\
### Training Loop

`train_detector` trains for `cfg.epochs` epochs with batch-level multi-scale and evaluates the EMA weights on the hold-out every `cfg.eval_interval` epochs; with `select = "last"` the EMA weights of the last epoch are the model (the `*` marks the evaluation whose weights were saved).

- `<run folder>/final.pt` contains the model weights (EMA, last epoch).
- `<run folder>/final-last.pt` contains the latest state (model, optimizer, EMA, history); re-running this cell resumes from it.
"""

TRAIN = """\
def log(message: str) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\\n")


model, history = train_detector(
    cfg=cfg,
    data_root=data_root,
    checkpoint_dir=det_checkpoint_dir,
    run_name=run_name,
    device=device,
    log=log,
    smoke=smoke,
)
"""

CURVES_MD = """\
### Training Curves
"""

CURVES = """\
epochs = [record["epoch"] for record in history]
evaluated = [record for record in history if "AP" in record]

figure, axes = plt.subplots(1, 2, figsize=(14, 4))
for key in ("train_total", "train_box", "train_cls", "train_dfl"):
    axes[0].plot(epochs, [record[key] for record in history], label=key.removeprefix("train_"))
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("training loss")
axes[0].legend()
for key, label in (("AP", "AP"), ("AP50", "AP50"), ("AP_small", "AP small"), ("AP_large", "AP large")):
    axes[1].plot([record["epoch"] for record in evaluated], [record[key] for record in evaluated], marker="o", label=label)
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("hold-out AP (EMA weights)")
axes[1].legend()
plt.tight_layout()
plt.show()
"""

SUBMISSION_MD = """\
### Kaggle Submission (final detector)

Load the final checkpoint, measure the hold-out AP at `eval_image_size` without and with test-time augmentation, and write `result.csv` for all test images with the augmented setting (this file replaces the YOLOv3 one written above).

Predictions use original-image coordinates, class-wise NMS, and at most 100 boxes per image.
"""

SUBMISSION = """\
model = build_model(cfg, pretrained=False).to(device).eval()
model.load_state_dict(torch.load(det_checkpoint_dir / f"{run_name}.pt", map_location="cpu", weights_only=True))

settings = {
    "plain": {"hflip_tta": False, "tta_scales": ()},
    f"hflip + scales {tta_scales}": {"hflip_tta": True, "tta_scales": tta_scales},
}
for name, tta in settings.items():
    metrics = evaluate_detector(
        model,
        val_loader,
        annotation_data=val_annotations,
        category_ids=val_eval_dataset.category_ids,
        conf_thres=cfg.conf_thres,
        nms_thres=cfg.nms_thres,
        **tta,
    )
    print(
        f"hold-out @{eval_image_size} {name}: AP {metrics['AP']:.4f} | AP50 {metrics['AP50']:.4f}"
        f" | AP75 {metrics['AP75']:.4f} | APs {metrics['AP_small']:.4f} | APm {metrics['AP_medium']:.4f}"
        f" | APl {metrics['AP_large']:.4f}"
    )
print("per-class AP: " + ", ".join(f"{k} {v:.3f}" for k, v in metrics["per_class_ap"].items() if v is not None))

result_path = write_submission(
    model,
    test_loader,
    output_path=RESULT_PATH,
    conf_thres=cfg.conf_thres,
    nms_thres=cfg.nms_thres,
    hflip_tta=True,
    tta_scales=tta_scales,
)
"""


def main() -> int:
    if TARGET.exists() and "--force" not in sys.argv:
        print(f"refusing to overwrite {TARGET}; pass --force or give another output path", file=sys.stderr)
        return 1
    nb = json.loads(TEMPLATE.read_text())
    cells = nb["cells"]

    def cell(i: int, must_contain: str) -> dict:
        c = cells[i]
        src = "".join(c["source"])
        assert must_contain in src, f"cell {i} does not contain {must_contain!r}"
        return c

    set_source(cell(2, "Hi I'm XXX"), "**Your Answer:**    \nHi I'm 黃毓峰, 315834011.")
    cell(6, "from src.yolo import getODmodel")
    cell(22, "from src.predict_test import predict_test")
    assert len(cells) == 23, len(cells)

    tail = [
        markdown(APPROACH_MD),
        code(IMPORTS),
        markdown(CONFIG_MD),
        code(CONFIG),
        markdown(DATASET_MD),
        code(DATASET),
        markdown(INSPECT_MD),
        code(INSPECT),
        markdown(MODEL_MD),
        code(MODEL),
        markdown(LOSS_MD),
        code(LOSS),
        markdown(TRAIN_MD),
        code(TRAIN),
        markdown(CURVES_MD),
        code(CURVES),
        markdown(SUBMISSION_MD),
        code(SUBMISSION),
    ]
    cells.extend(tail)

    TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {TARGET} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
