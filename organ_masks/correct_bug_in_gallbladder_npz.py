#!/usr/bin/env python3
"""
Fix NPZ joint masks: bladder = bladder \ gallbladder, then keep only largest component (bladder only).

Normal mode
-----------
- Overwrite *_gt.npz in-place (preserve NPZ key and dtype).
- Optional label subset via --output_mask_names (alphabetically sorted).

Debug mode (--debug)
--------------------
- Process only 10 items (after filtering/partitioning), do NOT overwrite NPZ.
- Save each label as separate NIfTI: {root}_debugging/<rel>/<ID>/segmentations/<label>.nii.gz
- Also save CT as NIfTI if {ID}.npz exists: {root}_debugging/<rel>/<ID>/ct.nii.gz
  (identity geometry; spacing = (1,1,1))

Class handling
--------------
- --input_mask_names: YAML of classes present in NPZ (we sort alphabetically to define channel order).
- 'gall_bladder'/'gallbladder' normalized to 'gallbladder' internally; same for 'bladder'.

Partitioning
------------
- --parts P and --part N split a deterministically sorted file list for multi-machine runs.

Examples
--------
python fix_npz_bladder.py --root /data/npz --input_mask_names classes.yaml
python fix_npz_bladder.py --root /data/npz --input_mask_names classes.yaml --output_mask_names keep.yaml
python fix_npz_bladder.py --root /data/npz --input_mask_names classes.yaml --parts 4 --part 2
python fix_npz_bladder.py --root /data/npz --input_mask_names classes.yaml --ids ids.csv --workers 24
python fix_npz_bladder.py --root /data/npz --input_mask_names classes.yaml --debug
"""

import argparse
import os
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict

import numpy as np
from tqdm import tqdm

# Prefer scipy.ndimage for CC; fall back to scikit-image
try:
    from scipy import ndimage as ndi
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
    try:
        from skimage.measure import label as sk_label
        _HAVE_SKIMAGE = True
    except Exception:
        _HAVE_SKIMAGE = False

def require_cc():
    if not _HAVE_SCIPY and not _HAVE_SKIMAGE:
        raise RuntimeError(
            "Connected-component labeling requires either scipy or scikit-image.\n"
            "Please install one of them, e.g.: pip install scipy   (recommended)"
        )

def read_yaml_list(path: Path) -> List[str]:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("pyyaml not installed. Please: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        names = data
    elif isinstance(data, dict):
        if "classes" in data and isinstance(data["classes"], list):
            names = data["classes"]
        else:
            lists = [v for v in data.values() if isinstance(v, list)]
            if not lists:
                raise ValueError(f"YAML at {path} does not contain a list of class names.")
            names = lists[0]
    else:
        raise ValueError(f"YAML at {path} must define a list or a dict with a list.")
    names = [str(x).strip() for x in names if str(x).strip()]
    names = sorted(names, key=lambda s: s.casefold())
    return names

def normalize_name(s: str) -> str:
    s = s.lower()
    for ch in [" ", "_", "-"]:
        s = s.replace(ch, "")
    return s

def canonical_label_filename(name: str) -> str:
    """
    'gallbladder' -> 'gall_bladder.nii.gz'; others: lowercase, spaces/hyphens -> underscores.
    """
    nn = normalize_name(name)
    if nn == "gallbladder":
        base = "gall_bladder"
    else:
        base = name.lower().replace(" ", "_").replace("-", "_")
    return f"{base}.nii.gz"

def find_channel_indices(input_names: List[str]) -> Tuple[Optional[int], Optional[int]]:
    norm = [normalize_name(x) for x in input_names]
    bladder_idx = None
    gall_idx = None
    for i, n in enumerate(norm):
        if n == "bladder":
            bladder_idx = i
        elif n == "gallbladder":
            gall_idx = i
    return bladder_idx, gall_idx

def load_npz_array(path: Path) -> Tuple[np.ndarray, str]:
    with np.load(str(path)) as data:
        key = "arr_0" if "arr_0" in data.files else data.files[0]
        arr = data[key]
    return arr, key

def save_npz_array(path: Path, arr: np.ndarray, key: str):
    np.savez_compressed(str(path), **{key: arr})

def ensure_channel_first(arr: np.ndarray, n_classes: int) -> np.ndarray:
    if arr.ndim < 4:
        raise ValueError(f"Expected at least 4D array (C,Z,Y,X). Got shape {arr.shape}")
    if arr.shape[0] == n_classes:
        return arr
    if arr.shape[-1] == n_classes:
        axes = list(range(arr.ndim))
        axes = [arr.ndim - 1] + axes[:-1]
        return np.transpose(arr, axes)
    raise ValueError(
        f"Could not infer channel axis. Array shape {arr.shape} does not match n_classes={n_classes} "
        f"on first or last dimension."
    )

def largest_component_3d(binary: np.ndarray) -> np.ndarray:
    require_cc()
    if not binary.any():
        return binary
    if _HAVE_SCIPY:
        structure = np.ones((3,) * binary.ndim, dtype=int)
        labeled, n = ndi.label(binary, structure=structure)
        if n <= 1:
            return binary
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest = counts.argmax()
        return labeled == largest
    else:
        labeled = sk_label(binary, connectivity=3)
        if labeled.max() <= 1:
            return binary
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest = counts.argmax()
        return labeled == largest

# ---------- Debug saving helpers (NIfTI writing) ----------

def _require_sitk():
    try:
        import SimpleITK as sitk
        return sitk
    except ImportError as e:
        raise RuntimeError("--debug requested but SimpleITK is not installed. Please: pip install SimpleITK") from e

def _to_3d_volume(arr: np.ndarray) -> np.ndarray:
    """
    Convert CT array to 3D (Z,Y,X).
    - If 4D with a singleton channel (first/last), squeeze it.
    - If 4D with multiple channels, take the first channel.
    """
    a = arr
    if a.ndim == 3:
        return a
    if a.ndim == 4:
        # Try to squeeze singleton channel dim
        if a.shape[0] == 1:
            return a[0]
        if a.shape[-1] == 1:
            return a[..., 0]
        # Fall back to first channel (assume channel-first)
        return a[0]
    # Attempt to squeeze any singleton dims
    a = np.squeeze(a)
    if a.ndim == 3:
        return a
    raise ValueError(f"Cannot interpret CT array of shape {arr.shape} as a 3D volume.")

def save_labels_as_niis_debug(
    arr_czyx: np.ndarray,
    label_names: List[str],
    root: Path,
    npz_path: Path,
    debug_root: Path,
):
    """
    Save each channel as separate NIfTI:
      {debug_root}/<rel_parent_to_root>/<ID>/segmentations/<label>.nii.gz
    Geometry: identity (spacing=1).
    """
    sitk = _require_sitk()
    if arr_czyx.ndim != 4:
        raise ValueError(f"Expected (C,Z,Y,X) array. Got: {arr_czyx.shape}")

    rel_parent = npz_path.parent.relative_to(root)
    file_id = id_from_filename(npz_path) or npz_path.stem.replace("_gt", "")
    out_dir = debug_root / rel_parent / file_id / "segmentations"
    out_dir.mkdir(parents=True, exist_ok=True)

    for c, name in enumerate(label_names):
        vol = (arr_czyx[c] > 0).astype(np.uint8)
        img = sitk.GetImageFromArray(vol)
        img.SetSpacing((1.0, 1.0, 1.0))
        out_name = canonical_label_filename(name)
        sitk.WriteImage(img, str(out_dir / out_name))

def save_ct_as_nii_debug(
    root: Path,
    npz_gt_path: Path,
    debug_root: Path,
) -> bool:
    """
    If {ID}.npz exists next to {ID}_gt.npz, save it as NIfTI:
      {debug_root}/<rel_parent_to_root>/<ID>/ct.nii.gz
    Returns True if saved, False if {ID}.npz not found.
    """
    sitk = _require_sitk()

    file_id = id_from_filename(npz_gt_path) or npz_gt_path.stem.replace("_gt", "")
    ct_npz_path = npz_gt_path.with_name(f"{file_id}.npz")
    if not ct_npz_path.exists():
        return False

    ct_arr, _ = load_npz_array(ct_npz_path)
    ct_vol = _to_3d_volume(ct_arr)

    rel_parent = npz_gt_path.parent.relative_to(root)
    out_dir = debug_root / rel_parent / file_id
    out_dir.mkdir(parents=True, exist_ok=True)

    img = sitk.GetImageFromArray(ct_vol.astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(out_dir / "ct.nii.gz"))
    return True

# ---------- Core processing ----------

def process_one_npz(
    npz_path_str: str,
    input_names_sorted: List[str],
    output_names_sorted: Optional[List[str]] = None,
    debug: bool = False,
    root: Optional[Path] = None,
    debug_root: Optional[Path] = None,
) -> Tuple[str, str, str]:
    """
    Worker: edits bladder channel; either overwrites NPZ (normal) or writes per-label + CT NIfTIs (debug).
    Status: 'ok', 'ok_no_gall', 'error'
    detail: 'debug_saved', 'debug_saved_with_ct', 'debug_saved_no_ct', or error message.
    """
    p = Path(npz_path_str)
    try:
        arr, key = load_npz_array(p)
        C_in = len(input_names_sorted)
        arr = ensure_channel_first(arr, C_in)
        dtype = arr.dtype

        bladder_idx, gall_idx = find_channel_indices(input_names_sorted)
        if bladder_idx is None:
            return str(p), "error", "No 'bladder' class found in input_mask_names."

        # Bladder +/- gallbladder
        bladder = arr[bladder_idx] > 0
        used_gall = False
        if gall_idx is not None:
            gall = arr[gall_idx] > 0
            bladder = np.logical_and(bladder, np.logical_not(gall))
            used_gall = True

        # Largest CC on bladder only
        bladder = largest_component_3d(bladder)

        # Update bladder channel
        arr[bladder_idx] = bladder.astype(dtype)

        # Determine output array + names
        if output_names_sorted is not None:
            input_map: Dict[str, int] = {normalize_name(n): i for i, n in enumerate(input_names_sorted)}
            out_indices, out_names = [], []
            for name in output_names_sorted:
                nn = normalize_name(name)
                if nn not in input_map:
                    return str(p), "error", f"Requested output label '{name}' not found in input class list."
                out_indices.append(input_map[nn])
                out_names.append(name)
            arr_out = arr[out_indices]
            names_out = out_names
        else:
            arr_out = arr
            names_out = input_names_sorted

        if debug:
            if root is None or debug_root is None:
                return str(p), "error", "Internal: debug paths not provided."
            # Save labels as NIfTI
            save_labels_as_niis_debug(arr_out, names_out, root, p, debug_root)
            # Save CT as NIfTI if available
            saved_ct = save_ct_as_nii_debug(root, p, debug_root)
            return str(p), ("ok" if used_gall else "ok_no_gall"), ("debug_saved_with_ct" if saved_ct else "debug_saved_no_ct")
        else:
            # Overwrite NPZ
            save_npz_array(p, arr_out, key)
            return str(p), ("ok" if used_gall else "ok_no_gall"), ""

    except Exception as e:
        return str(p), "error", f"{type(e).__name__}: {e}"

def slice_for_part(n_items: int, part: int, parts: int) -> Tuple[int, int]:
    chunk = math.ceil(n_items / parts)
    start = part * chunk
    end = min(n_items, (part + 1) * chunk)
    return start, end

def read_id_list(ids_path: Path) -> set:
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

def id_from_filename(path: Path) -> Optional[str]:
    name = path.name
    if not name.endswith("_gt.npz"):
        return None
    return name[:-7]  # strip '_gt.npz'

def main():
    ap = argparse.ArgumentParser(description="Fix joint NPZ masks: bladder minus gallbladder, then largest component (bladder only).")
    ap.add_argument("--root", required=True, type=str, help="Root folder to search recursively for *_gt.npz.")
    ap.add_argument("--input_mask_names", required=True, type=str, help="YAML with ALL class names present in NPZ (sorted alphabetically).")
    ap.add_argument("--output_mask_names", type=str, default=None,
                    help="Optional YAML with class names to SAVE (subset). Sorted alphabetically before saving.")
    ap.add_argument("--ids", type=str, default=None,
                    help="Optional TXT/CSV with IDs; only process files named {ID}_gt.npz when ID is in the list.")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel workers (default: CPU count).")
    ap.add_argument("--dry-run", action="store_true", help="List target files but do not modify.")
    ap.add_argument("--part", type=int, default=0, help="Partition index (0-based).")
    ap.add_argument("--parts", type=int, default=1, help="Total number of partitions.")
    ap.add_argument("--debug", action="store_true",
                    help="Process only 10 items; save per-label NIfTIs and CT as NIfTI under {root}_debugging.")
    args = ap.parse_args()

    if args.parts <= 0:
        raise SystemExit("--parts must be >= 1")
    if not (0 <= args.part < args.parts):
        raise SystemExit(f"--part must be in [0, {args.parts - 1}]")
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    # Load class lists (sorted)
    input_names_sorted = read_yaml_list(Path(args.input_mask_names))
    output_names_sorted = read_yaml_list(Path(args.output_mask_names)) if args.output_mask_names else None

    # Build candidate list
    candidates = list(root.rglob("*_gt.npz"))

    # Optional ID filter
    if args.ids:
        allowed = read_id_list(Path(args.ids))
        candidates = [p for p in candidates if (id_from_filename(p) in allowed)]

    if not candidates:
        print("No *_gt.npz files found after filtering.")
        return

    # Deterministic ordering
    candidates = sorted(candidates, key=lambda p: str(p))

    # Partition
    start, end = slice_for_part(len(candidates), args.part, args.parts)
    targets = candidates[start:end]

    # Debug: cap at 10
    if args.debug:
        targets = targets[:10]

    print(f"Total found: {len(candidates)} | parts={args.parts} | processing part={args.part} "
          f"[{start}:{end}] -> {len(targets)} files"
          + (" [DEBUG: capped at 10]" if args.debug else ""))

    if args.dry_run:
        for p in targets:
            print(p)
        return

    require_cc()  # early dependency check

    debug_root = Path(str(root) + "_debugging") if args.debug else None
    if args.debug:
        debug_root.mkdir(parents=True, exist_ok=True)

    stats = {"ok": 0, "ok_no_gall": 0, "error": 0}
    details = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                process_one_npz,
                str(p),
                input_names_sorted,
                output_names_sorted,
                args.debug,
                root if args.debug else None,
                debug_root if args.debug else None,
            )
            for p in targets
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            path_str, status, detail = fut.result()
            stats[status] = stats.get(status, 0) + 1
            if detail:
                details.append((path_str, status, detail))

    print("\nSummary:")
    print(f"  OK (used gallbladder):      {stats.get('ok',0)}")
    print(f"  OK (no gallbladder class):  {stats.get('ok_no_gall',0)}")
    print(f"  Errors:                     {stats.get('error',0)}")

    if details:
        print("\nNotes / Errors:")
        for p, st, d in details:
            print(f"- [{st}] {p}: {d}")

if __name__ == "__main__":
    main()