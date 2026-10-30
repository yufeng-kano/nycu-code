"""Verify and extract the per-folder zips downloaded from the Kaggle Data page.

The competition's API cannot serve train/val files, so those folders are fetched
through the website's folder download, which yields flat zips named after the
folder (train.zip, train_mask.zip, val.zip, val_mask.zip, test.zip). This script
checks each archive (CRC, member names, sizes against the API manifest), extracts
it into data/<folder>/ and reports what is still missing.

Usage:
    python scripts/tools/extract_zips.py --data data            # all data/*.zip
    python scripts/tools/extract_zips.py --data data --zips ~/Downloads/train.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

FOLDERS: list[str] = ["train", "train_mask", "val", "val_mask", "test"]
MANIFEST_NAME: str = ".kaggle_manifest.json"


def load_manifest(data: Path) -> dict[str, int]:
    path = data / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(f"missing {path}; run scripts/tools/download_data.py once to create it")
    return {f["name"]: f["bytes"] for f in json.loads(path.read_text())}


def member_ok(name: str, folder: str, allowed: set[str]) -> bool:
    """Accept `<file>` or `<folder>/<file>` where <file> is a manifest entry of that folder."""
    parts = name.split("/")
    if len(parts) == 2 and parts[0] != folder:
        return False
    if len(parts) > 2 or any(part in ("", ".", "..") for part in parts):
        return False
    return parts[-1] in allowed


def folder_of(zip_path: Path) -> str:
    stem = zip_path.stem.lower()
    for folder in sorted(FOLDERS, key=len, reverse=True):
        if stem == folder or stem.startswith(folder + "-") or stem.startswith(folder + " ("):
            return folder
    raise SystemExit(f"cannot tell which folder {zip_path.name} holds; rename it to <folder>.zip")


def extract_one(zip_path: Path, data: Path, manifest: dict[str, int]) -> bool:
    folder = folder_of(zip_path=zip_path)
    allowed = {k.split("/")[-1] for k in manifest if k.startswith(folder + "/")}
    out_dir = data / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{zip_path.name} -> {out_dir}")

    with zipfile.ZipFile(zip_path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            print(f"  CRC error at {corrupt}; archive is damaged, download it again")
            return False
        members = [m for m in archive.infolist() if not m.is_dir()]
        odd = [m.filename for m in members if not member_ok(name=m.filename, folder=folder, allowed=allowed)]
        if odd:
            print(f"  {len(odd)} member names not in the manifest for {folder}/, refusing to extract: {odd[:5]}")
            return False
        extracted = skipped = mismatched = 0
        for member in members:
            name = member.filename.split("/")[-1]
            expected = manifest.get(f"{folder}/{name}")
            if expected is not None and expected != member.file_size:
                mismatched += 1
            target = out_dir / name
            if target.is_file() and target.stat().st_size == member.file_size:
                skipped += 1
                continue
            with archive.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted += 1
    expected_names = {k.split("/")[-1] for k in manifest if k.startswith(folder + "/")}
    present = {p.name for p in out_dir.glob("*.png")}
    missing = sorted(expected_names - present)
    print(f"  members {len(members)}, extracted {extracted}, already present {skipped}, size mismatches vs manifest {mismatched}")
    print(f"  folder now holds {len(present)} / {len(expected_names)} files; missing {len(missing)} {missing[:3]}")
    return mismatched == 0 and not missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--zips", type=Path, nargs="*", default=None, help="archives to process; default data/*.zip")
    args = parser.parse_args()
    data: Path = args.data.resolve()
    manifest = load_manifest(data=data)
    zips = args.zips if args.zips else sorted(data.glob("*.zip"))
    if not zips:
        print("no zip files found")
        return 1
    ok = all([extract_one(zip_path=z.resolve(), data=data, manifest=manifest) for z in zips])

    print("\nfolder status:")
    complete = True
    for folder in FOLDERS:
        expected = sum(1 for k in manifest if k.startswith(folder + "/"))
        present = len(list((data / folder).glob("*.png"))) if (data / folder).is_dir() else 0
        complete &= present == expected
        print(f"  {folder:11s} {present:6d} / {expected}")
    csv_ok = (data / "test_index.csv").is_file()
    print(f"  test_index.csv {'present' if csv_ok else 'MISSING'}")
    return 0 if ok and complete and csv_ok else 1


if __name__ == "__main__":
    sys.exit(main())
