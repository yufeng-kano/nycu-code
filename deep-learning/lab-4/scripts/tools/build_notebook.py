"""Build HW4_315834011.ipynb from the TA template by filling every TODO block.

The template file is never modified. Cells are addressed by index after
verifying a distinctive substring, so a template change fails loudly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
TEMPLATE = LAB / "Lab4_Semantic_Segmentation_on_CIHP.ipynb"
# Output path: argv[1] if given, else the HW4 notebook. An existing output is never overwritten unless --force is passed,
# because the executed HW4 notebook holds the training record the TA grades.
TARGET = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else LAB / "HW4_315834011.ipynb"


def replace_block(src: str, start_marker: str, end_marker: str, body: str) -> str:
    """Replace the text between the line containing start_marker and the line containing end_marker."""
    lines = src.split("\n")
    starts = [i for i, l in enumerate(lines) if start_marker in l]
    ends = [i for i, l in enumerate(lines) if end_marker in l]
    assert len(starts) == 1, (start_marker, len(starts))
    after = [e for e in ends if e > starts[0]]
    assert after, end_marker
    s, e = starts[0], after[0]
    if set(lines[e - 1].strip()) == {"#"}:  # keep the top border of the "End of your code" box
        e -= 1
    return "\n".join(lines[: s + 1] + body.rstrip("\n").split("\n") + lines[e:])


def set_source(cell: dict, src: str) -> None:
    cell["source"] = src.splitlines(keepends=True)


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

    # --- 2: name + SID -------------------------------------------------------
    set_source(cell(2, "Hi I'm XXX"), "**Your Answer:**    \nHi I'm 黃毓峰, 315834011.")

    # --- 9: output locations follow the lab-3 layout --------------------------
    c = cell(9, 'OUTPUT_DIR = DATA_ROOT.parent / "outputs"')
    src = "".join(c["source"]).replace(
        'OUTPUT_DIR = DATA_ROOT.parent / "outputs"\nOUTPUT_DIR.mkdir(exist_ok=True)',
        'OUTPUT_DIR = DATA_ROOT.parent / "checkpoints"  # model weights (lab-3 layout)\n'
        'OUTPUT_DIR.mkdir(exist_ok=True)\n'
        'CSV_DIR = DATA_ROOT.parent  # output.csv at the lab root',
    )
    set_source(c, src)

    # --- 16: augmentation + loaders -----------------------------------------
    c = cell(16, "transforms_train = ...")
    src = "".join(c["source"])
    src = replace_block(
        src,
        "#  This one is for training, find the composition to get better result #",
        "#                           End of your code                           #",
        '''########################################################################
PAD_COLOR: tuple[int, int, int] = (124, 116, 104)  # grey border of the prepared images (= ImageNet mean)

transforms_train = A.Compose([
    A.Affine(
        scale=(0.7, 1.3),
        translate_percent=(-0.1, 0.1),
        rotate=(-10, 10),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_CONSTANT,
        fill=PAD_COLOR,
        fill_mask=IGNORE_INDEX,
        p=0.7,
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=10,
        p=0.5,
    ),
])''',
    )
    src = replace_block(
        src,
        "#  For validation/test we avoid strong augmentations that change labels #",
        "#                           End of your code                           #",
        '''########################################################################
transforms_val = A.Compose([])''',
    )
    src = src.replace(
        "batch_size= 8\n\ntrain_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)\n"
        "val_loader = DataLoader(val_set, batch_size=1, shuffle=False)",
        "batch_size = 16\nnum_workers = 8\n\n"
        "train_loader = DataLoader(\n"
        "    dataset=train_set,\n"
        "    batch_size=batch_size,\n"
        "    shuffle=True,\n"
        "    num_workers=num_workers,\n"
        "    pin_memory=True,\n"
        "    persistent_workers=True,\n"
        "    drop_last=True,\n"
        ")\n"
        "val_loader = DataLoader(\n"
        "    dataset=val_set,\n"
        "    batch_size=1,\n"
        "    shuffle=False,\n"
        "    num_workers=4,\n"
        "    pin_memory=True,\n"
        "    persistent_workers=True,\n"
        ")",
    )
    assert "num_workers=num_workers" in src
    set_source(c, src)

    # --- 19: DoubleConv / Down / Up -----------------------------------------
    c = cell(19, "class DoubleConv(nn.Module):")
    src = "".join(c["source"])
    src = src.replace(
        "        self.double_conv = nn.Sequential(\n            ...\n        )",
        "        self.double_conv = nn.Sequential(\n"
        "            nn.Conv2d(\n"
        "                in_channels=in_channels,\n"
        "                out_channels=mid_channels,\n"
        "                kernel_size=3,\n"
        "                padding=1,\n"
        "                bias=False,\n"
        "            ),\n"
        "            nn.BatchNorm2d(num_features=mid_channels),\n"
        "            nn.ReLU(inplace=True),\n"
        "            nn.Conv2d(\n"
        "                in_channels=mid_channels,\n"
        "                out_channels=out_channels,\n"
        "                kernel_size=3,\n"
        "                padding=1,\n"
        "                bias=False,\n"
        "            ),\n"
        "            nn.BatchNorm2d(num_features=out_channels),\n"
        "            nn.ReLU(inplace=True),\n"
        "        )",
    )
    src = src.replace(
        "        self.maxpool_conv = nn.Sequential(\n            ...\n        )",
        "        self.maxpool_conv = nn.Sequential(\n"
        "            nn.MaxPool2d(kernel_size=2),\n"
        "            DoubleConv(in_channels=in_channels, out_channels=out_channels),\n"
        "        )",
    )
    src = src.replace(
        "        x1 = ...\n        diffY = ...\n        diffX = ...\n        x1 = ...\n        x = ...",
        "        x1 = self.up(x1)\n"
        "        diffY = x2.size(2) - x1.size(2)\n"
        "        diffX = x2.size(3) - x1.size(3)\n"
        "        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])\n"
        "        x = torch.cat([x2, x1], dim=1)",
    )
    assert "..." not in src.replace("...", "", 0) or src.count("...") == 0, "unfilled TODO in cell 19"
    set_source(c, src)

    # --- 20: UNet -----------------------------------------------------------
    c = cell(20, "class UNet(nn.Module):")
    src = "".join(c["source"])
    src = src.replace(
        "        self.down1 = ...\n        self.down2 = ...\n        self.down3 = ...\n        self.down4 = ...\n"
        "        self.up1 = ...\n        self.up2 = ...\n        self.up3 = ...\n        self.up4 = ...\n        self.outc = ...",
        "        self.down1 = Down(in_channels=64, out_channels=128)\n"
        "        self.down2 = Down(in_channels=128, out_channels=256)\n"
        "        self.down3 = Down(in_channels=256, out_channels=512)\n"
        "        self.down4 = Down(in_channels=512, out_channels=1024)\n"
        "        self.up1 = Up(in_channels=1024, out_channels=512)\n"
        "        self.up2 = Up(in_channels=512, out_channels=256)\n"
        "        self.up3 = Up(in_channels=256, out_channels=128)\n"
        "        self.up4 = Up(in_channels=128, out_channels=64)\n"
        "        self.outc = OutConv(in_channels=64, out_channels=n_classes)",
    )
    src = src.replace(
        "        x2 = ...\n        x3 = ...\n        x4 = ...\n        x5 = ...\n"
        "        x = ...\n        x = ...\n        x = ...\n        x = ...\n        logits = ...",
        "        x2 = self.down1(x1)\n"
        "        x3 = self.down2(x2)\n"
        "        x4 = self.down3(x3)\n"
        "        x5 = self.down4(x4)\n"
        "        x = self.up1(x5, x4)\n"
        "        x = self.up2(x, x3)\n"
        "        x = self.up3(x, x2)\n"
        "        x = self.up4(x, x1)\n"
        "        logits = self.outc(x)",
    )
    assert src.count("...") == 0, "unfilled TODO in cell 20"
    set_source(c, src)

    # --- 24: pixel accuracy --------------------------------------------------
    c = cell(24, "def pixel_accuracy(")
    src = "".join(c["source"])
    src = replace_block(
        src,
        "# TODO: argmax over classes; compute accuracy only at valid pixels.",
        "#                                      End of your code",
        "        #############################################################################################################\n"
        "        pred = output.argmax(dim=1)\n"
        "        correct = (pred[valid] == mask[valid]).sum()\n"
        "        accuracy = float(correct.item() / valid.sum().item())",
    )
    assert "        ...\n" not in src
    set_source(c, src)

    # --- 26: mIoU ------------------------------------------------------------
    c = cell(26, "def mIoU(")
    src = "".join(c["source"])
    src = src.replace(
        "            intersect = ...\n            union = ...",
        "            intersect = (pred_c & label_c).sum()\n            union = (pred_c | label_c).sum()",
    )
    assert src.count("...") == 0
    set_source(c, src)

    # --- 28: Dice loss -------------------------------------------------------
    c = cell(28, "class DiceLoss(nn.Module):")
    src = "".join(c["source"])
    src = replace_block(
        src,
        "# TODO: implement masked soft Dice loss.",
        "#          End of your code           #",
        "        #######################################\n"
        "        n_classes = inputs.shape[1]\n"
        "        probs = F.softmax(inputs.float(), dim=1)\n"
        "        valid = (targets != self.ignore_index).unsqueeze(1).float()  # (B, 1, H, W)\n"
        "        safe_targets = torch.where(targets == self.ignore_index, torch.zeros_like(targets), targets)\n"
        "        one_hot = F.one_hot(safe_targets, num_classes=n_classes).permute(0, 3, 1, 2).float()\n"
        "        probs = probs * valid\n"
        "        one_hot = one_hot * valid\n"
        "        intersection = (probs * one_hot).sum(dim=(0, 2, 3))\n"
        "        cardinality = (probs + one_hot).sum(dim=(0, 2, 3))\n"
        "        dice = (2.0 * intersection + eps) / (cardinality + eps)  # one value per class",
    )
    assert "        ...\n" not in src
    set_source(c, src)

    # --- 30: fit -------------------------------------------------------------
    c = cell(30, "def fit(epochs, model, train_loader, val_loader, criterion1, criterion2, optimizer, scheduler, patch=False):")
    src = "".join(c["source"])
    src = replace_block(
        src,
        "# Also implement zero_grad(), backward() and optimizer.step() in this block.",
        "#                           End of your code                           #",
        "            image = image.to(device, non_blocking=True)\n"
        "            mask = mask.to(device, non_blocking=True)\n"
        "            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):\n"
        "                output = model(image)\n"
        "            loss = 0.5 * criterion1(output.float(), mask) + 0.5 * criterion2(output, mask)\n"
        "            iou_score += mIoU(output, mask)\n"
        "            accuracy += pixel_accuracy(output, mask)\n"
        "            optimizer.zero_grad(set_to_none=True)\n"
        "            loss.backward()\n"
        "            optimizer.step()",
    )
    src = src.replace(
        "                    image, mask = data\n\n"
        "                    image = ...\n"
        "                    mask = ...\n"
        "                    output = ...\n"
        "                    val_iou_score += ...\n"
        "                    test_accuracy += ...\n"
        "                    loss = ...\n"
        "                    test_loss += ...",
        "                    image, mask = data\n\n"
        "                    image = image.to(device, non_blocking=True)\n"
        "                    mask = mask.to(device, non_blocking=True)\n"
        "                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):\n"
        "                        output = model(image)\n"
        "                    val_iou_score += mIoU(output, mask)\n"
        "                    test_accuracy += pixel_accuracy(output, mask)\n"
        "                    loss = 0.5 * criterion1(output.float(), mask) + 0.5 * criterion2(output, mask)\n"
        "                    test_loss += loss.item()",
    )
    # model selection by val mIoU (the graded metric) instead of val loss
    src = src.replace(
        "    min_loss = np.inf\n    not_improve = 0\n",
        "    best_miou = 0.0\n    not_improve = 0\n"
        "    use_amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()\n",
    )
    src = src.replace(
        "            if (test_loss / len(val_loader)) < min_loss:\n"
        "                print('Loss Decreasing.. {:.3f} >> {:.3f}'.format(min_loss, test_loss / len(val_loader)))\n"
        "                min_loss = test_loss / len(val_loader)\n"
        "                not_improve = 0\n"
        "                torch.save(model.state_dict(), OUTPUT_DIR / 'Unet-CIHP-best.pt')\n"
        "            else:\n"
        "                not_improve += 1\n"
        "                print(f'Loss Not Decrease for {not_improve} time')\n",
        "            epoch_val_miou = val_iou_score / len(val_loader)\n"
        "            if epoch_val_miou > best_miou:\n"
        "                print('Val mIoU improving.. {:.4f} >> {:.4f}'.format(best_miou, epoch_val_miou))\n"
        "                best_miou = epoch_val_miou\n"
        "                not_improve = 0\n"
        "                torch.save(model.state_dict(), OUTPUT_DIR / 'Unet-CIHP-best.pt')\n"
        "            else:\n"
        "                not_improve += 1\n"
        "                print(f'Val mIoU not improving for {not_improve} epochs')\n",
    )
    src = src.replace(
        "            if not_improve >= 7:\n                print('Loss not decrease for 7 epochs, Stop Training')\n",
        "            if not_improve >= 7:\n                print('Val mIoU not improving for 7 epochs, Stop Training')\n",
    )
    assert src.count("...") == 0, "unfilled TODO in fit"
    assert "best_miou" in src and "use_amp" in src
    set_source(c, src)

    # --- 31: hyper-parameters -------------------------------------------------
    c = cell(31, "max_lr = 1e-3")
    src = "".join(c["source"]).replace("epoch = 10\n", "epoch = 20\n")
    set_source(c, src)

    # --- 36: markdown path note ------------------------------------------------
    c = cell(36, "Export `outputs/output.csv`")
    set_source(c, "".join(c["source"]).replace("Export `outputs/output.csv`", "Export `output.csv` at the lab root"))

    # --- 43: csv location ------------------------------------------------------
    c = cell(43, "output_path = OUTPUT_DIR / 'output.csv'")
    set_source(c, "".join(c["source"]).replace("output_path = OUTPUT_DIR / 'output.csv'", "output_path = CSV_DIR / 'output.csv'"))

    # sanity: no TODO placeholder survives anywhere
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            assert " = ...\n" not in src and "\n        ...\n" not in src, f"placeholder left in cell {i}"
            c["outputs"] = []
            c["execution_count"] = None

    TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    print("wrote", TARGET, "cells", len(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
