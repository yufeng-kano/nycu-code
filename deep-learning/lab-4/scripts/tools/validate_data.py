"""Validate the downloaded CIHP course package against the Lab 4 spec.

Usage:
    python scripts/tools/validate_data.py --data data [--overlay result/overlay.png]

Checks: folder layout, per-split counts against the slides (28,280 / 1,000 /
3,961), image/mask pairing, PNG integrity, 320 x 320 size, RGB images,
single-channel masks with values in 0-19 or 255, test_index.csv consistency,
byte-identical duplicates across splits, class pixel histogram, and the
training-set RGB mean/std. With --overlay it also renders a few image+mask
overlays so pairing and alignment can be checked by eye.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

N_CLASSES: int = 20
IGNORE: int = 255
SIZE: tuple[int, int] = (320, 320)
EXPECTED: dict[str, int] = {"train": 28280, "val": 1000, "test": 3961}
SPLIT_DIRS: list[str] = ["train", "train_mask", "val", "val_mask", "test"]


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_image(path: Path) -> dict:
    """Return findings for one RGB image file."""
    out: dict = {"path": str(path), "errors": [], "md5": md5_of(path=path)}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            out["mode"] = image.mode
            arr = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
            out["sum"] = arr.sum(axis=(0, 1))
            out["sumsq"] = (arr**2).sum(axis=(0, 1))
            out["npix"] = arr.shape[0] * arr.shape[1]
            if image.size != SIZE:
                out["errors"].append(f"size {image.size}")
            if image.mode != "RGB":
                out["errors"].append(f"mode {image.mode}")
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(f"open failed: {exc!r}")
    return out


def check_mask(path: Path) -> dict:
    """Return findings for one mask file."""
    out: dict = {"path": str(path), "errors": [], "md5": md5_of(path=path)}
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            out["mode"] = image.mode
            arr = np.asarray(image)
            if arr.ndim != 2:
                out["errors"].append(f"ndim {arr.ndim} shape {arr.shape}")
                arr = arr.reshape(arr.shape[0], arr.shape[1], -1)[..., 0]
            if arr.dtype != np.uint8:
                out["errors"].append(f"dtype {arr.dtype}")
            if image.size != SIZE:
                out["errors"].append(f"size {image.size}")
            hist = np.bincount(arr.reshape(-1).astype(np.int64), minlength=256)
            out["hist"] = hist
            illegal = [int(v) for v in np.nonzero(hist)[0] if v >= N_CLASSES and v != IGNORE]
            if illegal:
                out["errors"].append(f"illegal values {illegal}")
            out["valid_pixels"] = int(hist[:N_CLASSES].sum())
            out["present"] = [int(c) for c in np.nonzero(hist[:N_CLASSES])[0]]
            if out["valid_pixels"] == 0:
                out["errors"].append("no valid pixel")
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(f"open failed: {exc!r}")
    return out


def list_pngs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"))


def report_errors(label: str, results: list[dict], limit: int = 10) -> int:
    bad = [r for r in results if r["errors"]]
    print(f"  {label}: {len(results)} files, {len(bad)} with problems")
    for r in bad[:limit]:
        print(f"    {Path(r['path']).name}: {r['errors']}")
    if len(bad) > limit:
        print(f"    ... {len(bad) - limit} more")
    return len(bad)


def check_layout(data: Path) -> int:
    problems = 0
    print(f"DATA DIR: {data}")
    for name in SPLIT_DIRS:
        exists = (data / name).is_dir()
        print(f"  {name}/: {'ok' if exists else 'MISSING'}")
        problems += 0 if exists else 1
        if exists:
            stray = [p.name for p in (data / name).iterdir() if p.suffix != ".png"]
            if stray:
                print(f"    stray non-png files: {stray[:5]} ({len(stray)})")
                problems += 1
    index_csv = data / "test_index.csv"
    print(f"  test_index.csv: {'ok' if index_csv.is_file() else 'MISSING'}")
    problems += 0 if index_csv.is_file() else 1
    extras = sorted(p.name for p in data.iterdir() if p.name not in SPLIT_DIRS + ["test_index.csv"] and not p.name.startswith("."))
    if extras:
        print(f"  other entries at root: {extras}")
    return problems


def check_test_index(data: Path) -> int:
    print("\n=== test_index.csv ===")
    index_csv = data / "test_index.csv"
    if not index_csv.is_file():
        return 1
    with index_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = list(rows[0].keys()) if rows else []
    print(f"  columns: {columns}, rows: {len(rows)}")
    indices = [int(r["index"]) for r in rows]
    names = [r["filename"] for r in rows]
    on_disk = [p.name for p in list_pngs(folder=data / "test")]
    ok_index = indices == list(range(len(rows)))
    ok_order = names == on_disk
    missing = sorted(set(names) - set(on_disk))
    extra = sorted(set(on_disk) - set(names))
    print(f"  columns == [index, filename]: {columns == ['index', 'filename']}")
    print(f"  index is 0..N-1 contiguous: {ok_index}")
    print(f"  filename order == sorted(test/*.png): {ok_order}")
    print(f"  listed but missing on disk: {len(missing)} {missing[:5]}")
    print(f"  on disk but not listed: {len(extra)} {extra[:5]}")
    bad = not (ok_index and ok_order) or bool(missing) or bool(extra) or columns != ["index", "filename"]
    return 1 if bad else 0


def render_overlay(data: Path, out: Path, per_split: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random.seed(0)
    rows: list[tuple[str, str]] = []
    for split in ["train", "val"]:
        stems = sorted(p.stem for p in (data / split).glob("*.png"))
        rows += [(split, s) for s in random.sample(population=stems, k=per_split)]
    fig, axes = plt.subplots(nrows=len(rows), ncols=3, figsize=(9, 3 * len(rows)))
    for (split, stem), ax in zip(rows, axes):
        image = Image.open(data / split / f"{stem}.png").convert("RGB")
        mask = np.asarray(Image.open(data / f"{split}_mask" / f"{stem}.png"))
        shown = np.ma.masked_equal(mask, IGNORE)
        ax[0].imshow(image)
        ax[0].set_title(f"{split}/{stem}", fontsize=8)
        ax[1].imshow(shown, cmap="tab20", vmin=0, vmax=N_CLASSES - 1, interpolation="nearest")
        ax[1].set_title(f"classes {sorted(np.unique(mask).tolist())}", fontsize=7)
        ax[2].imshow(image)
        ax[2].imshow(shown, alpha=0.55, cmap="tab20", vmin=0, vmax=N_CLASSES - 1, interpolation="nearest")
        ax[2].set_title("overlay", fontsize=8)
        for a in ax:
            a.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    print(f"\noverlay saved to {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--overlay", type=Path, default=None, help="write an image+mask overlay PNG here")
    args = parser.parse_args()
    data: Path = args.data.resolve()

    problems = check_layout(data=data)
    if problems:
        print(f"\nlayout problems: {problems}; stopping before file checks")
        return 2

    all_md5: dict[str, list[str]] = {}
    mask_hist_total = np.zeros(256, dtype=np.int64)
    img_sum = np.zeros(3)
    img_sumsq = np.zeros(3)
    img_npix = 0
    present_counter: Counter[int] = Counter()

    with ProcessPoolExecutor() as pool:
        for split in ["train", "val", "test"]:
            print(f"\n=== {split} ===")
            images = list_pngs(folder=data / split)
            print(f"  images: {len(images)} (expected {EXPECTED[split]})")
            problems += 0 if len(images) == EXPECTED[split] else 1
            image_results = list(pool.map(check_image, images, chunksize=64))
            problems += report_errors(label="image check", results=image_results)
            print(f"  image modes: {dict(Counter(r.get('mode') for r in image_results))}")
            for r in image_results:
                all_md5.setdefault(r["md5"], []).append(f"{split}/{Path(r['path']).name}")
                if "sum" in r and split == "train":
                    img_sum += r["sum"]
                    img_sumsq += r["sumsq"]
                    img_npix += r["npix"]
            if split == "test":
                continue

            masks = list_pngs(folder=data / f"{split}_mask")
            print(f"  masks: {len(masks)}")
            only_image = sorted({p.stem for p in images} - {p.stem for p in masks})
            only_mask = sorted({p.stem for p in masks} - {p.stem for p in images})
            if only_image or only_mask:
                problems += 1
                print(f"  PAIRING MISMATCH: {len(only_image)} images without mask {only_image[:5]}, "
                      f"{len(only_mask)} masks without image {only_mask[:5]}")
            else:
                print("  pairing: every image has a mask with the same stem")
            mask_results = list(pool.map(check_mask, masks, chunksize=64))
            problems += report_errors(label="mask check", results=mask_results)
            print(f"  mask modes: {dict(Counter(r.get('mode') for r in mask_results))}")
            ratios: list[float] = []
            background_only = 0
            for r in mask_results:
                all_md5.setdefault(r["md5"], []).append(f"{split}_mask/{Path(r['path']).name}")
                if "hist" in r:
                    mask_hist_total += r["hist"]
                    present_counter.update(r["present"])
                    ratios.append(r["valid_pixels"] / (SIZE[0] * SIZE[1]))
                    background_only += int(r["present"] == [0])
            if ratios:
                arr = np.asarray(ratios)
                print(f"  valid-pixel ratio per mask: min {arr.min():.3f} mean {arr.mean():.3f} max {arr.max():.3f}")
            print(f"  masks with only background: {background_only}")

    problems += check_test_index(data=data)

    print("\n=== byte-identical duplicates ===")
    groups = {k: v for k, v in all_md5.items() if len(v) > 1}
    cross = [v for v in groups.values() if len({x.split('/')[0].replace('_mask', '') for x in v}) > 1]
    print(f"  duplicate groups: {len(groups)}; spanning two splits: {len(cross)}")
    for v in cross[:10]:
        print(f"    {v}")
    for v in [v for v in groups.values() if v not in cross][:5]:
        print(f"    (within one split) {v}")

    print("\n=== class pixel share over train+val masks ===")
    total_valid = mask_hist_total[:N_CLASSES].sum()
    print(f"  ignore(255) share of all mask pixels: {mask_hist_total[IGNORE] / mask_hist_total.sum():.4f}")
    for c in range(N_CLASSES):
        print(f"  class {c:2d}: {mask_hist_total[c] / total_valid:8.4%} of valid pixels, present in {present_counter[c]} masks")

    if img_npix:
        mean = img_sum / img_npix
        std = np.sqrt(img_sumsq / img_npix - mean**2)
        print(f"\nTRAIN RGB mean {mean.round(4).tolist()} std {std.round(4).tolist()}")

    if args.overlay is not None:
        render_overlay(data=data, out=args.overlay.resolve())

    print(f"\nTOTAL PROBLEMS: {problems}")
    return 0 if problems == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
