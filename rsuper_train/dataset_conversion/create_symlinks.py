#!/usr/bin/env python3
"""
Create a symlinked subset of a BDMAP dataset quickly.

Directory-link mode (default):    dst/BDMAP_0000123 -> src/BDMAP_0000123
File-link    mode:                recreate tree with symlinks for each file.

Usage
-----
python make_subset_links.py \
       --src  /path/to/full_dataset \
       --dst  /path/to/subset_dataset \
       --csv  ids.csv \
       [--mode dir|file] [--workers 32]
"""
import argparse, os, re, sys, csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm

import pandas as pd

# ───────────────────────────── helpers ─────────────────────────────
def norm(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")     # 'BDMAP ID'→'bdmap_id'

def load_ids(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, engine="c")
    df.rename(columns={c: norm(c) for c in df.columns}, inplace=True)
    if "bdmap_id" not in df.columns:
        sys.exit(f"No “BDMAP ID” column in {csv_path}")
    return df["bdmap_id"].unique().tolist()

# fast, idempotent link ------------------------------------------------------
def safe_symlink(src: Path, dst: Path, dir_: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src, target_is_directory=dir_)
    except FileExistsError:
        pass

# per-case handlers ----------------------------------------------------------
def link_case_dir(src_root: Path, dst_root: Path, bid: str):
    src = src_root / bid
    dst = dst_root / bid
    if src.is_dir():
        safe_symlink(src, dst, dir_=True)
    elif (f := src.with_suffix(".nii.gz")).is_file():
        safe_symlink(f, dst_root / f.name, dir_=False)
    else:
        return bid   # missing
    return None

def link_case_files(src_root: Path, dst_root: Path, bid: str):
    src_case  = src_root / bid
    dst_case  = dst_root / bid
    missing   = None

    if src_case.is_dir():
        # walk once with scandir (much faster than os.walk + listdir)
        for root, dirs, files in os.walk(src_case):
            rel = Path(root).relative_to(src_case)
            tgt_root = dst_case / rel
            tgt_root.mkdir(parents=True, exist_ok=True)
            with os.scandir(root) as it:
                for entry in it:
                    if entry.is_file():
                        safe_symlink(Path(root) / entry.name,
                                     tgt_root / entry.name,
                                     dir_=False)
    elif (f := src_case.with_suffix(".nii.gz")).is_file():
        safe_symlink(f, dst_root / f.name, dir_=False)
    else:
        missing = bid
    return missing

# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path,
                    help="CSV with a “BDMAP ID” column")
    ap.add_argument("--mode", choices=["dir", "file"], default="file",
                    help="dir = 1 symlink per case (fastest, default); "
                         "file = replicate tree with per-file links")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Threads for I/O (only used in --mode file)")
    args = ap.parse_args()

    ids       = load_ids(args.csv)
    args.dst.mkdir(parents=True, exist_ok=True)

    # choose strategy
    if args.mode == "dir":
        missing = []
        for bid in tqdm(ids, desc="Linking (directory symlinks)", unit="case"):
            if link_case_dir(args.src, args.dst, bid):
                missing.append(bid)
    else:  # file mode – use a ThreadPool for per-file linking
        missing = []
        func = partial(link_case_files, args.src, args.dst)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for res in tqdm(pool.map(func, ids, chunksize=64),
                            total=len(ids), unit="case",
                            desc="Linking (per-file)"):
                if res:
                    missing.append(res)

    # summary ----------------------------------------------------------------
    if missing:
        print(f"\n⚠  {len(missing):,} IDs were NOT found in {args.src}:")
        for m in missing[:10]:
            print("   •", m)
        if len(missing) > 10:
            print("   …")

if __name__ == "__main__":
    main()