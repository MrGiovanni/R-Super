#!/usr/bin/env python3
# extract_lesions.py
from __future__ import annotations

import argparse
import shutil
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# YAML helpers                                                                #
# --------------------------------------------------------------------------- #
def dump_yaml(path: Path, data: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        yaml.safe_dump(list(data), path.open("w"),
                       default_flow_style=False, sort_keys=False)
    except ModuleNotFoundError:
        with path.open("w") as f:
            for x in data:
                f.write(f"- {x}\n")


def load_yaml(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        import yaml
        return set(yaml.safe_load(path.read_text()) or [])
    except ModuleNotFoundError:
        return {ln.strip("- \n") for ln in path.read_text().splitlines()
                if ln.startswith("-")}


# --------------------------------------------------------------------------- #
# lesion extraction                                                           #
# --------------------------------------------------------------------------- #
def extract_lesion_candidates(
    prob: np.ndarray,
    n_lesions: int,
    peak_cut: float = 0.40,
    min_voxels: int = 11,
    min_peak: float = 0.01,
) -> Tuple[np.ndarray, int]:
    out = np.zeros(prob.shape, dtype=np.uint8)
    work = prob.copy()
    conn = np.ones((3, 3, 3), dtype=np.uint8)
    kept = 0
    while kept < n_lesions:
        peak_val = work.max()
        if peak_val < min_peak:
            break
        peak_idx = np.unravel_index(work.argmax(), work.shape)
        lbl, _ = label(work >= peak_cut * peak_val, structure=conn)
        comp = lbl == lbl[peak_idx]
        if comp.sum() >= min_voxels:
            out[comp] = 1
            kept += 1
        work[comp] = 0.0
    return out, kept


# --------------------------------------------------------------------------- #
# helpers for empty folders                                                   #
# --------------------------------------------------------------------------- #
def copy_predictions_to_dest(src_bdmap: Path, dst_root: Path) -> None:
    """Copy *predictions/* directly into <dst_root>/<ID>/ (rename folder)."""
    pred_dir = src_bdmap / "predictions"
    dest_dir = dst_root / src_bdmap.stem        # strip ".npz"
    if pred_dir.exists():
        shutil.copytree(pred_dir, dest_dir, dirs_exist_ok=True)
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)  # create empty folder


# --------------------------------------------------------------------------- #
# per-BDMAP worker                                                            #
# --------------------------------------------------------------------------- #
def process_bdmap(
    bdmap_dir: str,
    meta_dict: dict[str, dict],
    out_nii_dir: str,
    out_npz_dir: str,
    input_root: str,
) -> Tuple[str, bool]:
    bdmap_path = Path(bdmap_dir)
    bdmap_id   = bdmap_path.stem            # drop ".npz"
    pred_raw_dir = bdmap_path / "predictions_raw"
    nii_files  = list(pred_raw_dir.rglob("*.nii.gz"))

    # ---------- empty (no predictions_raw nifti) ----------------------------
    if not nii_files:
        for root in (out_nii_dir, out_npz_dir):
            copy_predictions_to_dest(bdmap_path, Path(root))
        return bdmap_id, True

    masks_to_write: List[Tuple[str, np.ndarray, np.ndarray, nib.Nifti1Header]] = []
    included = True

    for nii_path in nii_files:
        organ = nii_path.stem.split("_")[0].lower()
        col   = f"number of {organ} lesion instances"
        if bdmap_id not in meta_dict or col not in meta_dict[bdmap_id]:
            included = False
            break
        try:
            n_lesions = int(meta_dict[bdmap_id][col])
        except (TypeError, ValueError):
            included = False
            break
        if n_lesions <= 0:
            continue                                    # no lesions required

        img  = nib.load(str(nii_path))
        prob = img.get_fdata(dtype=np.float32)
        if prob.ndim == 4:
            prob = prob[..., 0]

        mask, found = extract_lesion_candidates(prob, n_lesions)
        if found < n_lesions:
            included = False
            break

        masks_to_write.append((nii_path.name, mask, img.affine, img.header))

    # ---------- write masks if all organs satisfied -------------------------
    if included:
        for fname, mask, affine, header in masks_to_write:
            for i,root in enumerate([out_nii_dir, out_npz_dir],0):
                dest_dir = Path(root) / bdmap_id
                dest_dir.mkdir(parents=True, exist_ok=True)

                out_nii = dest_dir / fname
                out_npz = dest_dir / (Path(fname)
                                      .with_suffix("")
                                      .with_suffix(".npz").name)
                                      
                if i==0:
                    nib.save(nib.Nifti1Image(mask.astype(np.uint8),
                                            affine=affine, header=header),
                            str(out_nii))
                else:
                    np.savez_compressed(str(out_npz), mask=mask.astype(np.uint8))

    return bdmap_id, included


# --------------------------------------------------------------------------- #
# dataset slicing helper                                                      #
# --------------------------------------------------------------------------- #
def slice_indices(n: int, parts: int, part: int) -> slice:
    base = n // parts
    rem  = n % parts
    start = part * base + min(part, rem)
    stop  = start + base + (1 if part < rem else 0)
    return slice(start, stop)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",   required=True, type=Path)
    ap.add_argument("--metadata",    required=True, type=Path)
    ap.add_argument("--out_nii_dir", required=True, type=Path)
    ap.add_argument("--out_npz_dir", required=True, type=Path)
    ap.add_argument("--workers",     type=int, default=4)
    ap.add_argument("--parts",       type=int, default=1,
                    help="total number of dataset chunks")
    ap.add_argument("--part",        type=int, default=0,
                    help="index (0-based) of the chunk to process")
    args = ap.parse_args()

    if not 0 <= args.part < args.parts:
        raise ValueError("--part must be in [0, parts-1]")

    # ---- ensure output roots & list folders --------------------------------
    for root in (args.out_nii_dir, args.out_npz_dir):
        (root / "list").mkdir(parents=True, exist_ok=True)

    kept_yaml_nii  = args.out_nii_dir / "list/dataset.yaml"
    kept_yaml_npz  = args.out_npz_dir / "list/dataset.yaml"
    skip_yaml_nii  = args.out_nii_dir / "list/skipped.yaml"
    skip_yaml_npz  = args.out_npz_dir / "list/skipped.yaml"

    kept_ids    = load_yaml(kept_yaml_nii) | load_yaml(kept_yaml_npz)
    skipped_ids = load_yaml(skip_yaml_nii) | load_yaml(skip_yaml_npz)
    processed_ids = kept_ids | skipped_ids

    # ---- metadata ----------------------------------------------------------
    meta = pd.read_csv(args.metadata)
    id_col = "BDMAP_ID" if "BDMAP_ID" in meta.columns else "BDMAP ID"
    meta_dict = meta.set_index(id_col).to_dict(orient="index")

    # ---- discover <BDMAP_ID>.npz folders -----------------------------------
    all_dirs = [p for p in args.input_dir.glob("*.npz") if p.is_dir()]
    print(f'All dirs: {all_dirs}')
    if len(all_dirs)==0:
        all_dirs = [p for p in args.input_dir.glob("*.nii.gz") if p.is_dir()]
    if len(all_dirs)==0:
        raise ValueError('No output folder found')
    empty_dirs, non_empty_dirs = [], []
    for p in all_dirs:
        pr_dir = p / "predictions_raw"
        #print(f'Searched at: {pr_dir}')
        if not pr_dir.exists() or not any(pr_dir.glob("*.nii.gz")):
            empty_dirs.append(p)
        else:
            non_empty_dirs.append(p)

    print(f'Number of non-empty predictions_raw found: {len(non_empty_dirs)}')
    # ---- copy empty dirs immediately ---------------------------------------
    new_empty = [d for d in empty_dirs if d.stem not in processed_ids]
    for d in new_empty:
        for root in (args.out_nii_dir, args.out_npz_dir):
            copy_predictions_to_dest(d, Path(root))
        kept_ids.add(d.stem)

    # ---- build workload list ----------------------------------------------
    to_do = [d for d in non_empty_dirs if d.stem not in processed_ids]
    if not to_do:
        print("Nothing left to process.")
    else:
        to_do.sort(key=lambda x: x.stem)
        slice_dirs = to_do[slice_indices(len(to_do), args.parts, args.part)]

        print(f"Total non-empty folders pending: {len(to_do)}")
        print(f"Processing slice {args.part}/{args.parts} "
              f"→ {len(slice_dirs)} folders on this machine.")

        worker = partial(
            process_bdmap,
            meta_dict=meta_dict,
            out_nii_dir=str(args.out_nii_dir),
            out_npz_dir=str(args.out_npz_dir),
            input_root=str(args.input_dir),
        )

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(worker, str(b)): b.stem for b in slice_dirs}
            for fut in tqdm(as_completed(futs), total=len(futs),
                            unit="folder", desc="BDMAP"):
                bdmap_id, ok = fut.result()
                if ok:
                    kept_ids.add(bdmap_id)
                else:
                    skipped_ids.add(bdmap_id)

    # ----------------------------------------------------------------------- #
    # Re-scan output tree → rebuild YAML lists                                #
    # ----------------------------------------------------------------------- #
    all_present = {p.name for p in args.out_nii_dir.iterdir() if p.is_dir()}
    kept_ids.update(all_present)
    kept_ids.difference_update(skipped_ids)

    dump_yaml(kept_yaml_nii,  sorted(kept_ids))
    dump_yaml(kept_yaml_npz,  sorted(kept_ids))
    dump_yaml(skip_yaml_nii,  sorted(skipped_ids))
    dump_yaml(skip_yaml_npz,  sorted(skipped_ids))

    print(f"\n✅ kept: {len(kept_ids)}   ❌ skipped: {len(skipped_ids)}")
    print(f"Lists rebuilt from disk and saved under {kept_yaml_nii.parent}")


if __name__ == "__main__":
    main()