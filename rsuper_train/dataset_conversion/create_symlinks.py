#!/usr/bin/env python3
"""
Create a symlinked subset of a dataset.

Example
-------
python create_symlinks.py \
       --src /projects/bodymaps/Data/CT_RATE/ct_rate_nnunet_labels \
       --dst /projects/bodymaps/Data/CT_RATE/ct_rate_nnunet_labels_subset \
       --csv /projects/bodymaps/Data/CT_RATE/pancreas_and_some_normals.csv
"""
import argparse
import os
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------
def safe_symlink(src: Path, dst: Path) -> None:
    """Create dst -> src (parents auto-made); ignore if already exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
    except FileExistsError:
        pass


def link_case_directory(src_case: Path, dst_case: Path) -> None:
    """Mirror an entire case directory using symlinks."""
    for root, _, files in os.walk(src_case):
        rel = Path(root).relative_to(src_case)
        tgt_root = dst_case / rel
        for f in files:
            safe_symlink(Path(root) / f, tgt_root / f)


# ------------------------------------------------------------------
def main(src: str, dst: str, csv_path: str) -> None:
    src_root = Path(src).resolve()
    dst_root = Path(dst).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    # read IDs (strip spaces, ignore blank rows)
    df = pd.read_csv(csv_path, dtype=str)
    col = [c for c in df.columns if c.strip().lower() == "bdmap_id"]
    if not col:
        raise KeyError("CSV must contain a column named BDMAP_ID")
    ids = df[col[0]].dropna().str.strip().unique()

    missing = []
    for bid in ids:
        dir_case  = src_root / bid                  # /src/BDMAP_xxxxxxx/
        file_case = src_root / f"{bid}.nii.gz"      # /src/BDMAP_xxxxxxx.nii.gz
        dst_case  = dst_root / bid

        if dir_case.is_dir():
            link_case_directory(dir_case, dst_case)

        elif file_case.is_file():
            safe_symlink(file_case, dst_root / file_case.name)

        else:
            missing.append(bid)

    # report
    if missing:
        print(f"WARNING: {len(missing)} ID(s) were not found in {src_root}:")
        for m in missing:
            print("  •", m)


# ------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replicate a dataset with symlinks "
                                             "for the BDMAP_IDs listed in a CSV.")
    ap.add_argument("--src", required=True, help="Original dataset root")
    ap.add_argument("--dst", required=True, help="Destination root (symlinks)")
    ap.add_argument("--csv", required=True, help="CSV containing a BDMAP_ID column")
    args = ap.parse_args()

    main(args.src, args.dst, args.csv)
