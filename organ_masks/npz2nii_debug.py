#!/usr/bin/env python3
"""
Randomly select 5 NPZ cases and convert to NIfTI (labels + CT) into {root}_debugging.

- Finds cases via '{ID}_gt.npz' under --root (recursively).
- For each selected case:
    * Loads multi-channel mask NPZ (assumed channel-first: C,Z,Y,X).
    * If --input_mask_names YAML is provided, uses those names (alphabetically sorted) for channels.
      Otherwise, names are label_00, label_01, ...
    * Saves each channel as a separate NIfTI under:
        {root}_debugging/<relative path to npz parent>/<ID>/segmentations/<label>.nii.gz
    * If '{ID}.npz' exists next to '{ID}_gt.npz', saves CT as:
        {root}_debugging/<relative path>/<ID>/ct.nii.gz
- Identity geometry (spacing=1). Sources are NOT modified.

Examples:
    python npz_to_debug_niis.py --root /data/npz --input_mask_names classes.yaml
    python npz_to_debug_niis.py --root /data/npz --num 5 --seed 123
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

def read_yaml_list(path: Path) -> List[str]:
    import yaml
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

def load_npz_array(path: Path) -> Tuple[np.ndarray, str]:
    with np.load(str(path)) as data:
        key = "arr_0" if "arr_0" in data.files else data.files[0]
        arr = data[key]
    return arr, key

def ensure_channel_first(arr: np.ndarray, n_classes: int) -> np.ndarray:
    if arr.ndim < 4:
        raise ValueError(f"Expected at least 4D array (C,Z,Y,X). Got shape {arr.shape}")
    if arr.shape[0] == n_classes:
        return arr
    if arr.shape[-1] == n_classes:
        # move last axis to first
        axes = list(range(arr.ndim))
        axes = [arr.ndim - 1] + axes[:-1]
        return np.transpose(arr, axes)
    # Assume channel-first if unsure (dataset said channel-first)
    if arr.shape[0] == n_classes:
        return arr
    raise ValueError(
        f"Could not infer channel axis for shape {arr.shape} with n_classes={n_classes}."
    )

def normalize_name(s: str) -> str:
    s = s.lower()
    for ch in (" ", "_", "-"):
        s = s.replace(ch, "")
    return s

def canonical_label_filename(name: str) -> str:
    # 'gallbladder' -> 'gall_bladder.nii.gz'
    if normalize_name(name) == "gallbladder":
        base = "gall_bladder"
    else:
        base = name.lower().replace(" ", "_").replace("-", "_")
    return f"{base}.nii.gz"

def _require_sitk():
    try:
        import SimpleITK as sitk
        return sitk
    except ImportError as e:
        raise SystemExit("SimpleITK is required. Please install: pip install SimpleITK") from e

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
        if a.shape[0] == 1:
            return a[0]
        if a.shape[-1] == 1:
            return a[..., 0]
        return a[0]
    a = np.squeeze(a)
    if a.ndim == 3:
        return a
    raise ValueError(f"Cannot interpret CT array of shape {arr.shape} as a 3D volume.")

def id_from_gt(path: Path) -> Optional[str]:
    name = path.name
    return name[:-7] if name.endswith("_gt.npz") else None  # strip '_gt.npz'

def main():
    ap = argparse.ArgumentParser(description="Randomly pick 5 NPZ cases and convert labels + CT to NIfTI in {root}_debugging.")
    ap.add_argument("--root", required=True, type=str, help="Root folder to search recursively for *_gt.npz.")
    ap.add_argument("--input_mask_names", type=str, default=None,
                    help="Optional YAML with ALL class names present in NPZ. Sorted alphabetically for channel naming.")
    ap.add_argument("--num", type=int, default=5, help="Number of random cases to convert (default: 5).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0).")
    args = ap.parse_args()

    sitk = _require_sitk()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    # Collect candidates (cases by *_gt.npz)
    gt_files = sorted(root.rglob("*_gt.npz"), key=lambda p: str(p))
    if not gt_files:
        print("No *_gt.npz files found.")
        return

    # Random pick
    rnd = random.Random(args.seed)
    pick_n = min(args.num, len(gt_files))
    picks = rnd.sample(gt_files, k=pick_n)

    # Load label names (optional)
    if args.input_mask_names:
        label_names = read_yaml_list(Path(args.input_mask_names))
    else:
        label_names = None  # will generate generic names after inspecting data

    debug_root = Path(str(root) + "_debugging")
    debug_root.mkdir(parents=True, exist_ok=True)

    print(f"Selected {len(picks)} cases out of {len(gt_files)}. Writing to: {debug_root}")

    converted = 0
    for gt_path in picks:
        try:
            case_id = id_from_gt(gt_path) or gt_path.stem.replace("_gt", "")
            rel_parent = gt_path.parent.relative_to(root)

            # --- Load masks ---
            mask_arr, _ = load_npz_array(gt_path)

            # Determine class list & ensure channel-first
            if label_names is not None:
                names = label_names
                n_classes = len(names)
            else:
                # assume channel-first and derive number of classes
                if mask_arr.ndim < 4:
                    raise ValueError(f"Mask {gt_path} has shape {mask_arr.shape}, expected >=4D.")
                n_classes = mask_arr.shape[0]
                names = [f"label_{i:02d}" for i in range(n_classes)]

            mask_czyx = ensure_channel_first(mask_arr, n_classes)

            # --- Prepare output dirs ---
            out_dir_segs = debug_root / rel_parent / case_id / "segmentations"
            out_dir_case = debug_root / rel_parent / case_id
            out_dir_segs.mkdir(parents=True, exist_ok=True)
            out_dir_case.mkdir(parents=True, exist_ok=True)

            # --- Save each label as NIfTI ---
            for c, name in enumerate(names):
                vol = (mask_czyx[c] > 0).astype(np.uint8)
                img = sitk.GetImageFromArray(vol)
                img.SetSpacing((1.0, 1.0, 1.0))
                out_name = canonical_label_filename(name)
                sitk.WriteImage(img, str(out_dir_segs / out_name))

            # --- Save CT if present ---
            ct_npz = gt_path.with_name(f"{case_id}.npz")
            if ct_npz.exists():
                ct_arr, _ = load_npz_array(ct_npz)
                ct_vol = _to_3d_volume(ct_arr).astype(np.float32)
                ct_img = sitk.GetImageFromArray(ct_vol)
                ct_img.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(ct_img, str(out_dir_case / "ct.nii.gz"))
            else:
                # Not fatal; just inform
                pass

            converted += 1

        except Exception as e:
            print(f"[WARN] Failed on {gt_path}: {type(e).__name__}: {e}")

    print(f"Done. Converted {converted} case(s) to NIfTI under: {debug_root}")

if __name__ == "__main__":
    main()