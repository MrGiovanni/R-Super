#!/usr/bin/env python3
"""
Fix bladder masks by removing the gallbladder and keeping only the largest component.

Key options
-----------
--skip_no_gall_bladder / --no_skip_no_gall_bladder
    Default: skip masks without a gallbladder file beside them.
--part N --parts P
    Deterministically split work into P chunks and process only chunk N (0 <= N < P).
--ids PATH
    Filter to paths like .../{ID}/segmentations/bladder.nii.gz using BDMAP IDs from PATH.

Examples
--------
python fix_bladder_masks.py --root /data/segm --parts 4 --part 0
python fix_bladder_masks.py --root /data/segm --ids ids.csv --no_skip_no_gall_bladder
"""

import argparse
import os
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

def read_id_list(ids_path: Path):
    ids_path = Path(ids_path)
    if ids_path.suffix.lower() in {".txt", ".list"}:
        with open(ids_path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError("--ids provided but pandas is not installed. Please: pip install pandas") from e
    sep = "," if ids_path.suffix.lower() in {".csv"} else None
    df = pd.read_csv(ids_path, sep=sep)
    for col in ["BDMAP ID", "BDMAP_ID", "BDMAP Name", "BDMAP", "ID"]:
        if col in df.columns:
            return set(str(x) for x in df[col].dropna().astype(str).tolist())
    return set(str(x) for x in df.iloc[:, 0].dropna().astype(str).tolist())

def largest_component(binary_img: sitk.Image) -> sitk.Image:
    bin_u8 = sitk.Cast(binary_img > 0, sitk.sitkUInt8)
    cc = sitk.ConnectedComponent(bin_u8)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.BinaryThreshold(relabeled, 1, 1, 1, 0)
    return sitk.Cast(largest, sitk.sitkUInt8)

def subtract_and_cleanup(bladder_img: sitk.Image, gall_img: sitk.Image | None) -> sitk.Image:
    b_arr = sitk.GetArrayFromImage(bladder_img) > 0
    if gall_img is not None and bladder_img.GetSize() == gall_img.GetSize() \
       and bladder_img.GetDirection() == gall_img.GetDirection() \
       and bladder_img.GetSpacing() == gall_img.GetSpacing() \
       and bladder_img.GetOrigin() == gall_img.GetOrigin():
        g_arr = sitk.GetArrayFromImage(gall_img) > 0
        out_arr = np.logical_and(b_arr, np.logical_not(g_arr))
    else:
        out_arr = b_arr
    out_img = sitk.GetImageFromArray(out_arr.astype(np.uint8))
    out_img.CopyInformation(bladder_img)
    out_img = largest_component(out_img)
    return out_img

def process_one(bladder_path_str: str, gall_names=("gall_bladder.nii.gz", "gallbladder.nii.gz")):
    p = Path(bladder_path_str)
    try:
        bladder_img = sitk.ReadImage(str(p))
    except Exception as e:
        return str(p), "error", f"Read bladder failed: {e}"

    gall_img = None
    gall_path = None
    for candidate in gall_names:
        cand = p.with_name(candidate)
        if cand.exists():
            gall_path = cand
            break

    if gall_path is not None:
        try:
            gi = sitk.ReadImage(str(gall_path))
            if bladder_img.GetSize() == gi.GetSize():
                gall_img = gi
        except Exception:
            gall_img = None

    try:
        fixed = subtract_and_cleanup(bladder_img, gall_img)
        out_img = sitk.Cast(fixed, bladder_img.GetPixelID())
        sitk.WriteImage(out_img, str(p))
        if gall_img is not None:
            return str(p), "ok", ""
        else:
            return str(p), "ok_no_gall", ("gall not found or mismatch" if gall_path is None else "geometry mismatch")
    except Exception as e:
        return str(p), "error", f"Process/save failed: {e}"

def path_matches_id_rule(path_: Path, allowed_ids: set[str]) -> bool:
    if path_.name != "bladder.nii.gz":
        return False
    if path_.parent.name != "segmentations":
        return False
    id_dir = path_.parent.parent.name
    return id_dir in allowed_ids

def has_gall_in_folder(bladder_path: Path, gall_names=("gall_bladder.nii.gz", "gallbladder.nii.gz")) -> bool:
    folder = bladder_path.parent
    return any((folder / name).exists() for name in gall_names)

def slice_for_part(n_items: int, part: int, parts: int) -> tuple[int, int]:
    # contiguous block split on deterministically sorted list
    chunk = math.ceil(n_items / parts)
    start = part * chunk
    end = min(n_items, (part + 1) * chunk)
    return start, end

def main():
    ap = argparse.ArgumentParser(description="Fix bladder masks by subtracting gallbladder and keeping largest component.")
    ap.add_argument("--root", required=True, type=str, help="Root folder to search recursively.")
    ap.add_argument("--ids", type=str, default=None,
                    help="Optional CSV/TXT with BDMAP IDs. Only process .../{ID}/segmentations/bladder.nii.gz.")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel workers (default: CPU count).")
    ap.add_argument("--dry-run", action="store_true", help="List target files but do not modify.")
    # Skip behavior (default True)
    ap.add_argument("--skip_no_gall_bladder", dest="skip_no_gall_bladder", action="store_true",
                    help="Skip bladders that have no gallbladder file beside them (default).")
    ap.add_argument("--no_skip_no_gall_bladder", dest="skip_no_gall_bladder", action="store_false",
                    help="Process bladders even if no gallbladder file exists (fallback to largest component only).")
    ap.set_defaults(skip_no_gall_bladder=True)
    # Partitioning
    ap.add_argument("--part", type=int, default=0, help="Which partition index to process (0-based).")
    ap.add_argument("--parts", type=int, default=1, help="Total number of partitions.")
    args = ap.parse_args()

    if args.parts <= 0:
        raise SystemExit("--parts must be >= 1")
    if not (0 <= args.part < args.parts):
        raise SystemExit(f"--part must be in [0, {args.parts - 1}]")
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    allowed_ids = None
    if args.ids:
        allowed_ids = read_id_list(Path(args.ids))
        if not allowed_ids:
            raise SystemExit(f"No IDs loaded from {args.ids}")

    # Find all bladder.nii.gz recursively
    candidates = list(root.rglob("bladder.nii.gz"))

    # IDs filter
    if allowed_ids is not None:
        candidates = [p for p in candidates if path_matches_id_rule(p, allowed_ids)]

    # Skip if no gallbladder beside it (presence-only check)
    if args.skip_no_gall_bladder:
        candidates = [p for p in candidates if has_gall_in_folder(p)]

    if not candidates:
        print("No matching bladder.nii.gz files found after filtering.")
        return

    # Deterministic order for consistent partitioning across machines
    candidates = sorted(candidates, key=lambda p: str(p))

    # Partition
    start, end = slice_for_part(len(candidates), args.part, args.parts)
    targets = candidates[start:end]

    print(f"Total found: {len(candidates)} | parts={args.parts} | processing part={args.part} "
          f"[{start}:{end}] -> {len(targets)} files")
    if args.dry_run:
        for p in targets:
            print(p)
        return

    stats = {"ok": 0, "ok_no_gall": 0, "error": 0}
    details = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, str(p)) for p in targets]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            path_str, status, detail = fut.result()
            stats[status] = stats.get(status, 0) + 1
            if detail:
                details.append((path_str, status, detail))

    print("\nSummary:")
    print(f"  OK (with gall subtraction): {stats.get('ok',0)}")
    print(f"  OK (no gall / mismatch):    {stats.get('ok_no_gall',0)}")
    print(f"  Errors:                     {stats.get('error',0)}")

    if details:
        print("\nNotes / Errors:")
        for p, st, d in details:
            print(f"- [{st}] {p}: {d}")

if __name__ == "__main__":
    # Optionally limit ITK threads inside each process:
    # sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(4)
    main()