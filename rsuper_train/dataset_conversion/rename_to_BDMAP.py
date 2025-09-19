#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from tqdm import tqdm   # pip install tqdm

def load_mapping(mapping_csv: Path, invert: bool = False):
    """
    Load mapping CSV and be flexible about column names.

    If invert=True, swap the mapping (new_name ➜ original_name).
    """
    def _pick_col(fieldnames, candidates):
        lower = {c.lower(): c for c in fieldnames}
        for c in candidates:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    pairs = []
    with mapping_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Mapping CSV appears to have no header/columns.")

        headers = [h.strip() for h in reader.fieldnames]

        orig_col = _pick_col(
            headers,
            [
                "original_name_no_ext",
                "original",
                "old_name",
                "source",
                "Encrypted Accession Number",
                "Encrypted_Accession_Number",
            ],
        )
        new_col = _pick_col(
            headers,
            [
                "new_name_no_ext",
                "new",
                "target",
                "BDMAP_ID",
                "BDMAP ID",
                "bdmap_id",
            ],
        )

        if not orig_col or not new_col:
            raise ValueError(
                "Mapping CSV must include columns for ORIGINAL and NEW names.\n"
                f"  Got columns: {headers}"
            )

        for row in reader:
            orig = (row.get(orig_col, "") or "").strip()
            new  = (row.get(new_col, "") or "").strip()
            if not orig or not new:
                raise ValueError(f"Invalid row in mapping (empty value): {row}")
            if invert:
                pairs.append((new, orig))  # swap!
            else:
                pairs.append((orig, new))

    # Duplicate checks
    seen_orig, seen_new = set(), set()
    for o, n in pairs:
        if o in seen_orig:
            raise ValueError(f"Duplicate source name in mapping: {o}")
        if n in seen_new:
            raise ValueError(f"Duplicate target name in mapping: {n}")
        seen_orig.add(o)
        seen_new.add(n)

    return pairs

def rename_by_mapping(input_folder: Path, mapping_pairs, mapping_path_for_log: Path):
    successes, missing, collisions = [], [], []
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

    for old_dir, new_dir, orig, new in tqdm(planned, total=len(planned), desc="Renaming by mapping"):
        old_dir.rename(new_dir)
        successes.append((orig, new))

    with mapping_path_for_log.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["original_name_no_ext", "new_name_no_ext"])
        writer.writerows(successes)

    total_pairs = len(mapping_pairs)
    print("\nSummary (mapping mode)")
    print(f"  Total entries in mapping : {total_pairs}")
    print(f"  Applied renames          : {len(successes)}")
    print(f"  Missing source folders   : {len(missing)}")
    print(f"  Skipped (target exists)  : {len(collisions)}")

def rename_auto(input_folder: Path, init_bdmap: int, mapping_path: Path):
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
                ext = "".join(old_path.suffixes)
                original_base = old_path.name[:-len(ext)] if ext else old_path.stem
                writer.writerow([original_base, new_stem])

def rename_masks(input_folder: Path, mapping_pairs, mapping_path_for_log: Path):
    """Special mode: rename only the immediate subfolders of input_folder using mapping."""
    successes, missing, collisions = [], [], []
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

    for old_dir, new_dir, orig, new in tqdm(planned, total=len(planned), desc="Renaming masks"):
        old_dir.rename(new_dir)
        successes.append((orig, new))

    with mapping_path_for_log.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["original_name_no_ext", "new_name_no_ext"])
        writer.writerows(successes)

    total_pairs = len(mapping_pairs)
    print("\nSummary (masks mode)")
    print(f"  Total entries in mapping : {total_pairs}")
    print(f"  Applied renames          : {len(successes)}")
    print(f"  Missing source folders   : {len(missing)}")
    print(f"  Skipped (target exists)  : {len(collisions)}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rename to BDMAP_{index}. "
            "If --mapping is provided, rename existing folders by that mapping. "
            "If --masks is also set, restrict renaming to top-level subfolders. "
            "If --invert_mapping is set, swap mapping direction (new ➜ original)."
        )
    )
    parser.add_argument("--input_folder", required=True, type=Path,
                        help="Folder to process")
    parser.add_argument("--init_bdmap",   required=False, type=int, default=1,
                        help="Starting integer for the BDMAP index (auto mode only)")
    parser.add_argument("--csv_out",      default="bdmap_mapping.csv", type=Path,
                        help="Output CSV mapping file")
    parser.add_argument("--mapping",      type=Path,
                        help="CSV mapping file (flexible headers).")
    parser.add_argument("--masks", action="store_true",
                        help="When set with --mapping, apply renaming only to top-level subfolders (mask mode).")
    parser.add_argument("--invert_mapping", action="store_true",
                        help="Swap mapping direction (new ➜ original).")
    args = parser.parse_args()

    input_folder = args.input_folder.resolve()
    csv_out      = args.csv_out.resolve()

    if not input_folder.is_dir():
        raise SystemExit(f"Not a directory: {input_folder}")

    if args.mapping:
        mapping_csv = args.mapping.resolve()
        pairs = load_mapping(mapping_csv, invert=args.invert_mapping)
        if args.masks:
            rename_masks(input_folder, pairs, csv_out)
            print(f"\nDone! Applied {'inverted ' if args.invert_mapping else ''}mask-folder mapping "
                  f"from {mapping_csv} and wrote log to {csv_out}")
        else:
            rename_by_mapping(input_folder, pairs, csv_out)
            print(f"\nDone! Applied {'inverted ' if args.invert_mapping else ''}mapping "
                  f"from {mapping_csv} and wrote log to {csv_out}")
    else:
        rename_auto(input_folder, args.init_bdmap, csv_out)
        print(f"\nDone! Mapping saved to {csv_out}")

if __name__ == "__main__":
    main()