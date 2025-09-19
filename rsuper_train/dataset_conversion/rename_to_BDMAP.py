#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from tqdm import tqdm   # pip install tqdm

def load_mapping(mapping_csv: Path):
    """
    Load mapping CSV with header: original_name_no_ext,new_name_no_ext
    Returns a list of (original, new) pairs in file order.
    """
    pairs = []
    with mapping_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"original_name_no_ext", "new_name_no_ext"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Mapping CSV must have columns {required}, got {reader.fieldnames}"
            )
        for row in reader:
            orig = (row["original_name_no_ext"] or "").strip()
            new  = (row["new_name_no_ext"] or "").strip()
            if not orig or not new:
                raise ValueError(f"Invalid row in mapping: {row}")
            pairs.append((orig, new))
    # Basic duplicate checks
    seen_orig = set()
    seen_new  = set()
    for o, n in pairs:
        if o in seen_orig:
            raise ValueError(f"Duplicate original_name in mapping: {o}")
        if n in seen_new:
            raise ValueError(f"Duplicate new_name in mapping: {n}")
        seen_orig.add(o); seen_new.add(n)
    return pairs

def rename_by_mapping(input_folder: Path, mapping_pairs, mapping_path_for_log: Path):
    """
    Rename *directories* in input_folder according to mapping_pairs.
    - Missing source folders are SKIPPED (counted and reported).
    - Existing target folders are SKIPPED to avoid overwrite (counted and reported).
    Writes CSV of successfully applied renames to mapping_path_for_log.
    """
    successes = []
    missing   = []
    collisions = []

    # Plan ops, but skip missing/collisions
    planned = []
    for orig, new in mapping_pairs:
        old_dir = input_folder / orig
        new_dir = input_folder / new
        if not old_dir.exists() or not old_dir.is_dir():
            missing.append(orig)
            continue
        if new_dir.exists():
            collisions.append((orig, new))
            continue
        planned.append((old_dir, new_dir, orig, new))

    # Apply planned renames
    for old_dir, new_dir, orig, new in tqdm(planned, total=len(planned), desc="Renaming by mapping"):
        old_dir.rename(new_dir)
        successes.append((orig, new))

    # Log what we actually did
    with mapping_path_for_log.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["original_name_no_ext", "new_name_no_ext"])
        writer.writerows(successes)

    # Summary
    total_pairs = len(mapping_pairs)
    print("\nSummary (mapping mode)")
    print(f"  Total entries in mapping : {total_pairs}")
    print(f"  Applied renames          : {len(successes)}")
    print(f"  Missing source folders   : {len(missing)}")
    print(f"  Skipped (target exists)  : {len(collisions)}")

def rename_auto(input_folder: Path, init_bdmap: int, mapping_path: Path):
    """
    Automatic mode (no --mapping):
      • If input_folder contains subfolders each with ct.nii.gz (i.e., {name}/ct.nii.gz),
        then ONLY RENAME the folder {name} ➜ BDMAP_XXXXXXXX (ct.nii.gz stays as-is).
      • Otherwise, flat files ➜ create BDMAP_XXXXXXXX/ct.nii.gz for each (original behavior).
    """
    folders_with_ct = sorted(
        [p for p in input_folder.iterdir() if p.is_dir() and (p / "ct.nii.gz").is_file()],
        key=lambda p: p.name
    )

    with mapping_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["original_name_no_ext", "new_name_no_ext"])

        if folders_with_ct:
            for offset, old_dir in tqdm(
                enumerate(folders_with_ct), total=len(folders_with_ct), desc="Renaming folders"
            ):
                new_stem = f"BDMAP_{init_bdmap + offset:08d}"
                new_dir  = input_folder / new_stem
                if new_dir.exists():
                    raise FileExistsError(f"{new_dir} already exists; aborting to avoid overwrite")
                old_name = old_dir.name
                old_dir.rename(new_dir)
                writer.writerow([old_name, new_stem])
        else:
            files = sorted(p for p in input_folder.iterdir() if p.is_file())
            for offset, old_path in tqdm(
                    enumerate(files), total=len(files), desc="Renaming files"
                ):
                new_stem = f"BDMAP_{init_bdmap + offset:08d}"
                target_dir = input_folder / new_stem
                target_dir.mkdir(exist_ok=False)
                new_path = target_dir / "ct.nii.gz"
                if new_path.exists():
                    raise FileExistsError(f"{new_path} already exists; aborting to avoid overwrite")
                old_path.rename(new_path)
                # Record mapping of base names (no extension) to BDMAP id
                ext = "".join(old_path.suffixes)
                original_base = old_path.name[:-len(ext)] if ext else old_path.stem
                writer.writerow([original_base, new_stem])

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rename to BDMAP_{index}. "
            "If --mapping is provided, rename existing folders by that mapping "
            "(useful for applying CT-to-BDMAP mapping to a masks folder)."
        )
    )
    parser.add_argument("--input_folder", required=True, type=Path,
                        help="Folder to process")
    parser.add_argument("--init_bdmap",   required=False, type=int, default=1,
                        help="Starting integer for the BDMAP index (auto mode only)")
    parser.add_argument("--csv_out",      default="bdmap_mapping.csv", type=Path,
                        help="Output CSV mapping file")
    parser.add_argument("--mapping",      type=Path,
                        help="CSV mapping file (columns: original_name_no_ext,new_name_no_ext). "
                             "If provided, only folders matching 'original_name_no_ext' are renamed "
                             "to the corresponding 'new_name_no_ext'. Missing are skipped.")
    args = parser.parse_args()

    input_folder = args.input_folder.resolve()
    csv_out      = args.csv_out.resolve()

    if not input_folder.is_dir():
        raise SystemExit(f"Not a directory: {input_folder}")

    if args.mapping:
        mapping_csv = args.mapping.resolve()
        pairs = load_mapping(mapping_csv)
        rename_by_mapping(input_folder, pairs, csv_out)
        print(f"\nDone! Applied mapping from {mapping_csv} and wrote log to {csv_out}")
    else:
        rename_auto(input_folder, args.init_bdmap, csv_out)
        print(f"\nDone! Mapping saved to {csv_out}")

if __name__ == "__main__":
    main()