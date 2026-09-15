#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dice + NSD evaluator for pancreatic_lesion masks
- Fixes:
  * GT glob uses 'segmentations' (plural)
  * Prediction map keys are the full 'PanTS_xxxxxxxx' (no slicing)
  * Uses 'predictions' vs 'predictions_raw' based on --confidence_th
  * Resamples prediction to GT grid (shape + affine) to prevent mismatches
  * Stable handling of empty masks for Dice/NSD
  * Optional CC pruning; can interpret threshold in voxels or mm^3
"""

import argparse, glob, os, sys, csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from filelock import FileLock
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage
from tqdm import tqdm
import torch
from monai.metrics import compute_surface_dice
import re

# --------------------------- CSV layout ------------------------------ #
# Beyond dice/nsd: absolute volumes (mm^3) and the TP/FP/FN split, so an
# over-segmenting model can be told apart from an under-segmenting one, and
# lesion-level detection can be separated from boundary accuracy.
CSV_COLS = [
    "case", "dice", "nsd",
    "gt_vol_mm3", "pred_vol_mm3", "vol_ratio", "vol_err_mm3",
    "tp_mm3", "fp_mm3", "fn_mm3",
    "precision", "recall",
    "gt_cc", "pred_cc", "gt_cc_hit", "detected",
    "vox_mm3", "error",
]

def write_row(row, csv_path):
    lock = FileLock(csv_path + ".lock")
    with lock:
        hdr = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            if hdr:
                w.writeheader()
            w.writerow(row)

# --------------------------- metrics --------------------------------- #
def dice_score(pred_bool: torch.Tensor, gt_bool: torch.Tensor) -> float:
    # pred_bool and gt_bool are boolean tensors of same shape
    tp = (pred_bool & gt_bool).sum().item()
    denom = pred_bool.sum().item() + gt_bool.sum().item()
    return (2.0 * tp) / (denom + 1e-8)

# --------------------------- helpers --------------------------------- #
def extract_caseid(path: Path) -> str | None:
    """Case ID = the folder holding the `predictions`/`predictions_raw` dir,
    i.e. two levels up from `<case>/<subdir>/pancreatic_lesion.nii.gz`.
    Dataset-agnostic (BDMAP_*, PanTrack_*, ...)."""
    case_dir = path.parent.parent
    return case_dir.name or None

def raw_array(nii):
    """Read a NIfTI's data without `get_fdata()`.

    `get_fdata()` upcasts to float64 (2 GB for a 512x512x972 volume) and caches
    that array on the image object, so a single case held ~11 GB across the
    several loads below. Masks here are uint8 with no scl_slope/scl_inter, so
    reading `dataobj` in its native dtype is exact and ~8x smaller. If the file
    does declare scaling, fall back to the scaled path.
    """
    slope, inter = nii.header.get_slope_inter()
    if (slope not in (None, 1.0)) or (inter not in (None, 0.0)):
        return nii.get_fdata(dtype=np.float32)
    return np.asanyarray(nii.dataobj)

def load_binary_img(path):
    nii = nib.load(path)
    data = (raw_array(nii) > 0.5).astype(np.uint8)
    return data, nii

def load_binary_union(paths):
    """Load each NIfTI in `paths`, threshold > 0.5, OR them together.
    Returns (union_uint8_array, nii_of_first_path). If a later mask's
    grid differs from the first, it is nearest-neighbor-resampled.
    """
    if isinstance(paths, str):
        paths = [paths]
    nii0 = nib.load(paths[0])
    union = (raw_array(nii0) > 0.5).astype(np.uint8)
    for p in paths[1:]:
        nii = nib.load(p)
        if (nii.shape != nii0.shape) or (not np.allclose(nii.affine, nii0.affine, atol=1e-3)):
            nii_res = resample_from_to(nii, nii0, order=0)
            data = raw_array(nii_res)
        else:
            data = raw_array(nii)
        union |= (data > 0.5).astype(np.uint8)
    return union, nii0

# Lesion subtypes whose union forms a "pancreatic_lesion" mask when no
# combined file is shipped. Filenames starting with '_' are intentionally
# excluded (legacy/duplicate masks in some AbdomenAtlasPro releases).
LESION_SUBTYPE_FILES = (
    "pancreatic_pdac.nii.gz",
    "pancreatic_pnet.nii.gz",
    "pancreatic_cyst.nii.gz",
)

def gt_id_candidates(cid):
    """GT folder names to try for a predicted case ID. Prediction folders may
    carry a trailing nnU-Net-style channel suffix (`_0000`) that the GT folder
    does not, e.g. pred `PanTrack_001_20220111_0000` -> GT `PanTrack_001_20220111`.
    """
    cands = [cid]
    stripped = re.sub(r"_\d{4}$", "", cid)
    if stripped != cid:
        cands.append(stripped)
    return cands

def resolve_gt_paths(gt_root, cid):
    """Return list of GT mask paths for case `cid`, or [] if none.
    Prefers a combined `pancreatic_lesion.nii.gz`; otherwise falls back
    to the union of available pdac/pnet/cyst subtype masks.
    Searches both `<gt_root>/<cid>/segmentations/` and `<gt_root>/<cid>/`.
    """
    base_dirs = [
        os.path.join(gt_root, c, sub)
        for c in gt_id_candidates(cid)
        for sub in ("segmentations", "")
    ]
    for base in base_dirs:
        combined = os.path.join(base, "pancreatic_lesion.nii.gz")
        if os.path.isfile(combined) and not os.path.basename(combined).startswith("_"):
            return [combined]
        found = []
        for fn in LESION_SUBTYPE_FILES:
            if fn.startswith("_"):
                continue
            p = os.path.join(base, fn)
            if os.path.isfile(p):
                found.append(p)
        if found:
            return found
    return []

def load_prob_img(path, thr):
    nii = nib.load(path)
    data = (raw_array(nii).astype(np.float32, copy=False) > float(thr)).astype(np.uint8)
    return data, nii

def cc_filter(mask, min_vox):
    if min_vox <= 0:
        return mask
    # Label only inside the foreground bounding box. A connected component
    # cannot extend beyond the foreground, so this is equivalent to labeling the
    # full volume -- but `ndimage.label` allocates an int32 array the size of its
    # input (1 GB for a 512x512x972 volume), which dominated peak memory.
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return mask
    sl = tuple(slice(int(c.min()), int(c.max()) + 1) for c in idx)
    sub = mask[sl]
    lab, n = ndimage.label(sub, np.ones((3, 3, 3), dtype=np.uint8))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    remove = sizes < min_vox
    remove[0] = False
    sub[remove[lab]] = 0
    mask[sl] = sub
    return mask

def crop_to_union_bbox(pred, gt, margin=2):
    """Crop both masks to the bounding box of (pred | gt), padded by `margin`.

    Dice counts only foreground voxels and NSD depends only on the two surfaces
    and the distances between them -- all of which lie inside this box -- so the
    metrics are unchanged. The margin keeps a background rim around every
    foreground voxel so the erosion-based edge extraction cannot mistake the
    array border for a surface. MONAI crops internally too, but only after the
    caller has already materialized full-volume tensors; cropping here is what
    avoids that cost. Returns the inputs unchanged if the union is empty.
    """
    union = pred | gt
    idx = np.where(union)
    if len(idx[0]) == 0:
        return pred, gt
    sl = []
    for ax, coords in enumerate(idx):
        lo = max(int(coords.min()) - margin, 0)
        hi = min(int(coords.max()) + margin + 1, union.shape[ax])
        sl.append(slice(lo, hi))
    sl = tuple(sl)
    return pred[sl], gt[sl]

def count_cc(mask):
    """Number of 26-connected components in a boolean mask.
    Labels inside the foreground bbox only (see cc_filter for why)."""
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return 0
    sl = tuple(slice(int(c.min()), int(c.max()) + 1) for c in idx)
    _, n = ndimage.label(mask[sl], np.ones((3, 3, 3), dtype=np.uint8))
    return int(n)

def count_gt_cc_hit(pred, gt):
    """How many GT lesion components the prediction touches at all.
    Lesion-level detection, independent of how well the boundary is traced."""
    idx = np.where(gt)
    if len(idx[0]) == 0:
        return 0
    sl = tuple(slice(int(c.min()), int(c.max()) + 1) for c in idx)
    lab, n = ndimage.label(gt[sl], np.ones((3, 3, 3), dtype=np.uint8))
    if n == 0:
        return 0
    hit = np.unique(lab[pred[sl] & (lab > 0)])
    return int(hit.size)

def make_row(cid, dice, nsd, pred, gt, spacing):
    """Assemble one CSV row of overlap + volume metrics.
    `pred`/`gt` are boolean arrays on the same grid; `spacing` is in mm.
    """
    vox_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    tp = int(np.count_nonzero(pred & gt))
    pred_n = int(np.count_nonzero(pred))
    gt_n = int(np.count_nonzero(gt))
    fp, fn = pred_n - tp, gt_n - tp

    gt_vol, pred_vol = gt_n * vox_mm3, pred_n * vox_mm3
    fmt = lambda v: f"{v:.4f}"
    return dict(
        case=cid,
        dice=fmt(dice) if dice is not None else "",
        nsd=fmt(nsd) if nsd is not None else "",
        gt_vol_mm3=f"{gt_vol:.1f}",
        pred_vol_mm3=f"{pred_vol:.1f}",
        # pred/gt volume ratio: >1 over-segmentation, <1 under-segmentation
        vol_ratio=fmt(pred_vol / gt_vol) if gt_n else "",
        vol_err_mm3=f"{pred_vol - gt_vol:.1f}",
        tp_mm3=f"{tp * vox_mm3:.1f}",
        fp_mm3=f"{fp * vox_mm3:.1f}",
        fn_mm3=f"{fn * vox_mm3:.1f}",
        precision=fmt(tp / pred_n) if pred_n else "",
        recall=fmt(tp / gt_n) if gt_n else "",
        gt_cc=count_cc(gt),
        pred_cc=count_cc(pred),
        gt_cc_hit=count_gt_cc_hit(pred, gt),
        detected=int(tp > 0),
        vox_mm3=f"{vox_mm3:.5f}",
        error="",
    )

def ensure_aligned_pred_to_gt(pred_nii, gt_nii):
    """Resample prediction to GT grid if shape or affine differ."""
    if (pred_nii.shape != gt_nii.shape) or (not np.allclose(pred_nii.affine, gt_nii.affine, atol=1e-3)):
        pred_res = resample_from_to(pred_nii, gt_nii, order=0)  # nearest neighbor for labels
        pred_np = (raw_array(pred_res) > 0.5).astype(np.uint8)
        return pred_np
    else:
        return (raw_array(pred_nii) > 0.5).astype(np.uint8)

# --------------------------- worker ---------------------------------- #
def eval_case(task):
    (
        cid, pred_p, gt_p, tol, cc_thr, cc_units_mm3, thr,
        csvp, skip_neg
    ) = task

    try:
        # No GT file on disk (--missing_means_empty): score against an empty GT.
        # Load the prediction on its own grid, prune components, then treat as a
        # tumor-negative case (Dice=1 if pred is empty, else 0).
        if gt_p is None:
            if abs(thr - 0.5) < 1e-8:
                pred_np, pred_nii = load_binary_img(pred_p)
            else:
                pred_np, pred_nii = load_prob_img(pred_p, thr)
            if cc_thr > 0:
                if cc_units_mm3:
                    ps = pred_nii.header.get_zooms()[:3]
                    vox_vol_mm3 = float(ps[0] * ps[1] * ps[2])
                    min_vox = int(np.ceil(cc_thr / max(vox_vol_mm3, 1e-8)))
                else:
                    min_vox = int(cc_thr)
                pred_np = cc_filter(pred_np, max(min_vox, 1))
            if skip_neg:
                return None
            dice = 1.0 if not pred_np.any() else 0.0
            ps = pred_nii.header.get_zooms()[:3]
            pb = pred_np.astype(bool)
            write_row(make_row(cid, dice, dice, pb, np.zeros_like(pb), ps), csvp)
            return None

        # Load GT (single combined mask, or union of pdac/pnet/cyst subtypes)
        gt_np, gt_nii = load_binary_union(gt_p)
        gt_spacing = gt_nii.header.get_zooms()[:3]

        # Load pred (binary vs prob)
        if abs(thr - 0.5) < 1e-8:
            pred_np_raw, pred_nii = load_binary_img(pred_p)
        else:
            pred_np_raw, pred_nii = load_prob_img(pred_p, thr)
        # Re-wrap the thresholded mask on the prediction's own grid. Reuse the
        # already-loaded `pred_nii` for affine/header instead of re-reading the
        # file twice, which used to materialize two more full-volume arrays.
        pred_nii = nib.Nifti1Image(pred_np_raw.astype(np.uint8), pred_nii.affine, pred_nii.header)

        # Align prediction grid to GT grid
        pred_np = ensure_aligned_pred_to_gt(pred_nii, gt_nii)

        # Connected components pruning
        if cc_thr > 0:
            if cc_units_mm3:
                # Interpret cc_thr as mm^3; convert to voxels using GT spacing
                vox_vol_mm3 = float(gt_spacing[0] * gt_spacing[1] * gt_spacing[2])
                min_vox = int(np.ceil(cc_thr / max(vox_vol_mm3, 1e-8)))
            else:
                # Interpret cc_thr directly as voxels
                min_vox = int(cc_thr)
            pred_np = cc_filter(pred_np, max(min_vox, 1))

        # Handle empty GT
        if not gt_np.any():
            if skip_neg:
                return None
            # If both empty → perfect; else bad
            dice = 1.0 if not pred_np.any() else 0.0
            nsd  = dice
            write_row(make_row(cid, dice, nsd, pred_np.astype(bool),
                               gt_np.astype(bool), gt_spacing), csvp)
            return None

        # If pred empty but GT non-empty → Dice=0, NSD=0 (avoid MONAI NaN)
        if not pred_np.any():
            write_row(make_row(cid, 0.0, 0.0, pred_np.astype(bool),
                               gt_np.astype(bool), gt_spacing), csvp)
            return None

        # Restrict to the padded union bounding box before building tensors: for a
        # lesion this is a tiny fraction of the volume, and it leaves both metrics
        # unchanged (see crop_to_union_bbox).
        pred_c, gt_c = crop_to_union_bbox(pred_np.astype(bool), gt_np.astype(bool))

        p = torch.from_numpy(np.ascontiguousarray(pred_c))
        g = torch.from_numpy(np.ascontiguousarray(gt_c))

        dsc_val = dice_score(p, g)

        # Surface Dice over foreground only
        nsd_val = compute_surface_dice(
            p.unsqueeze(0).unsqueeze(0).float(),
            g.unsqueeze(0).unsqueeze(0).float(),
            include_background=False,
            spacing=tuple(float(s) for s in gt_spacing),
            class_thresholds=[float(tol)]
        ).item()

        # Volume/overlap stats on the crop: identical to the full volume, since
        # every foreground voxel of both masks lies inside the union bbox.
        row = make_row(cid, dsc_val, nsd_val, pred_c, gt_c, gt_spacing)
        write_row(row, csvp)
        return row

    except Exception as e:
        row = {c: "" for c in CSV_COLS}
        row.update(case=cid, dice="nan", nsd="nan", error=str(e))
        write_row(row, csvp)
        return None

# --------------------------- CLI / main ------------------------------ #
def split_parts(lst, n, idx):
    base, extra = divmod(len(lst), n)
    start = idx * base + min(idx, extra)
    end   = start + base + (1 if idx < extra else 0)
    return lst[start:] if idx == n - 1 else lst[start:end]

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pred_root", required=True)
    p.add_argument("--gt_root",   required=True)
    p.add_argument("--out_csv",   required=True)
    p.add_argument("--case_glob", default="*",
                   help="Glob for case folders under --pred_root (e.g. 'BDMAP_*', 'PanTrack_*').")

    p.add_argument("--tolerance", type=float, default=3.0,
                   help="Surface tolerance (mm) for NSD (class_thresholds).")
    p.add_argument("--cc_post",   type=float, default=1.0,
                   help="Min size for CC pruning. Interpreted as voxels unless --cc_mm3 is set.")
    p.add_argument("--cc_mm3",    action="store_true",
                   help="Interpret --cc_post as mm^3 instead of voxels.")
    p.add_argument("--confidence_th", type=float, default=0.5,
                   help="Threshold for binarizing probability maps; "
                        "0.5 uses binary masks in 'predictions/', else uses 'predictions_raw/'")
    p.add_argument("--skip_negatives", action="store_true",
                   help="Ignore scans whose GT mask is empty")
    p.add_argument("--missing_means_empty", action="store_true",
                   help="Treat a predicted case that has NO GT lesion mask file on disk as "
                        "if its GT were an all-empty mask (a tumor-negative case), instead "
                        "of dropping it from the evaluation. Lets negatives that ship no "
                        "lesion file be scored (Dice=1 if the prediction is also empty, "
                        "else 0). Still honors --skip_negatives.")
    p.add_argument("--workers",   type=int, default=10)
    p.add_argument("--num_parts", type=int, default=1)
    p.add_argument("--part",      type=int, default=0)
    p.add_argument("--continue",  dest="cont", action="store_true",
                   help="Skip IDs already present in CSV")
    return p.parse_args()

def summarize(csv_path):
    """Print the average Dice over every row currently in the CSV."""
    if not os.path.exists(csv_path):
        return
    with FileLock(csv_path + ".lock"):
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    vals = []
    for r in rows:
        try:
            v = float(r.get("dice", ""))
        except (TypeError, ValueError):
            continue
        if v == v:                          # drop nan (rows that errored)
            vals.append(v)
    if vals:
        print(f"\nAverage Dice: {sum(vals) / len(vals):.4f}  (n={len(vals)})")


def main():
    a = parse_args()

    # ---- Build prediction map: include subdir by --confidence_th
    pred_subdir = "predictions" if abs(a.confidence_th - 0.5) < 1e-8 else "predictions_raw"
    pred_pattern = os.path.join(a.pred_root, a.case_glob, pred_subdir, "pancreatic_lesion.nii.gz")
    pred_files = glob.glob(pred_pattern)
    pred_map = {}
    for p in pred_files:
        cid = extract_caseid(Path(p))
        if cid:
            pred_map[cid] = p
    print(f"Number of prediction masks found: {len(pred_map)}  (dir={pred_subdir})")

    # ---- Build GT map by direct lookup of each predicted ID
    # (avoid globbing the entire GT root, which can hold millions of files).
    # Prefers a combined `pancreatic_lesion.nii.gz`; falls back to the union
    # of pdac + pnet + cyst subtype masks.
    gt_map = {}
    missing_gt = []
    n_combined = 0
    n_union = 0
    for cid in pred_map:
        paths = resolve_gt_paths(a.gt_root, cid)
        if not paths:
            missing_gt.append(cid)
            continue
        gt_map[cid] = paths
        if len(paths) == 1 and os.path.basename(paths[0]) == "pancreatic_lesion.nii.gz":
            n_combined += 1
        else:
            n_union += 1
    print(f"Number of GT masks found: {len(gt_map)} "
          f"(combined={n_combined}, subtype-union={n_union}; "
          f"predicted IDs without a GT mask: {len(missing_gt)})")
    if missing_gt:
        print("  sample missing GT IDs:", missing_gt[:5])

    # Optionally: score predicted IDs that have no GT file against an empty
    # (all-zero) GT, i.e. treat them as tumor-negative cases, rather than
    # silently dropping them. `None` is the sentinel eval_case reads.
    if a.missing_means_empty and missing_gt:
        for cid in missing_gt:
            gt_map[cid] = None
        print(f"  --missing_means_empty: treating {len(missing_gt)} predicted IDs "
              f"without a GT mask as empty (negative) GT.")

    # Intersect IDs
    ids = sorted(set(gt_map) & set(pred_map))
    if not ids:
        # Helpful debug dump (limited)
        sample_pred = dict(list(pred_map.items())[:5])
        print("No matching IDs. Sample pred_map keys:", sample_pred)
        sys.exit("No matching IDs")

    # --continue
    if a.cont and os.path.exists(a.out_csv):
        with FileLock(a.out_csv + ".lock"):
            done = set()
            with open(a.out_csv, "r") as f:
                # skip header, take first column as 'case'
                next(f, None)
                for line in f:
                    parts = line.strip().split(",")
                    if parts:
                        done.add(parts[0])
        ids = [i for i in ids if i not in done]
    else:
        with FileLock(a.out_csv + ".lock"):
            if os.path.exists(a.out_csv):
                os.remove(a.out_csv)

    if a.num_parts > 1:
        ids = split_parts(ids, a.num_parts, a.part)

    print(f"cases={len(ids)}  part={a.part}/{a.num_parts-1}  "
          f"workers={a.workers}  cc_post={a.cc_post}{' mm^3' if a.cc_mm3 else ' vox'}  "
          f"thr={a.confidence_th}  skip_negatives={a.skip_negatives}")

    tasks = [
        (
            cid,
            pred_map[cid],
            gt_map[cid],
            a.tolerance,
            a.cc_post,
            a.cc_mm3,
            a.confidence_th,
            a.out_csv,
            a.skip_negatives
        )
        for cid in ids
    ]

    if len(tasks) == 0:
        print("Nothing to do (all cases already evaluated?).")
        return

    if a.workers == 1:
        for t in tqdm(tasks, desc="eval"):
            eval_case(t)
    else:
        with Pool(a.workers) as pool:
            for _ in tqdm(pool.imap_unordered(eval_case, tasks),
                          total=len(tasks), desc="eval"):
                pass

    print("Finished – rows appended to", a.out_csv)
    summarize(a.out_csv)

if __name__ == "__main__":
    main()