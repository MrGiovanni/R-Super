#!/usr/bin/env python3
"""
Rename immediate subfolders of a directory by replacing the *suffix* 'PanTS'
with 'BDMAP'. Non-recursive. Skips if the target name already exists.

Usage:
  python rename_pants_to_bdmap.py /path/to/parent
  python rename_pants_to_bdmap.py /path/to/parent --dry-run
"""

from pathlib import Path
import argparse
import sys

SRC_SUFFIX = "PanTS"
DST_SUFFIX = "BDMAP"

def rename_subfolders(root_dir: Path, dry_run: bool = False) -> dict:
    """
    Rename immediate subfolders of `root_dir` whose names end with SRC_SUFFIX
    to end with DST_SUFFIX instead.

    Returns a dict with counts: {'renamed': int, 'skipped_no_match': int, 'skipped_exists': int, 'errors': int}
    """
    stats = {"renamed": 0, "skipped_no_match": 0, "skipped_exists": 0, "errors": 0}

    for p in root_dir.iterdir():
        if not p.is_dir():
            continue

        name = p.name
        if not name.endswith(SRC_SUFFIX):
            stats["skipped_no_match"] += 1
            continue

        new_name = name[: -len(SRC_SUFFIX)] + DST_SUFFIX
        target = p.with_name(new_name)

        if target.exists():
            print(f"[skip exists] {name} -> {new_name}", file=sys.stderr)
            stats["skipped_exists"] += 1
            continue

        if dry_run:
            print(f"[dry-run]     {name} -> {new_name}")
            stats["renamed"] += 1  # counts planned renames
            continue

        try:
            p.rename(target)
            print(f"[renamed]     {name} -> {new_name}")
            stats["renamed"] += 1
        except OSError as e:
            print(f"[error]       {name} -> {new_name}: {e}", file=sys.stderr)
            stats["errors"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Rename subfolders ending with 'PanTS' to 'BDMAP' (non-recursive).")
    parser.add_argument("root", type=Path, help="Path to the parent directory containing subfolders to rename.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without making changes.")
    args = parser.parse_args()

    root_dir = args.root
    if not root_dir.exists():
        print(f"Error: path does not exist: {root_dir}", file=sys.stderr)
        sys.exit(1)
    if not root_dir.is_dir():
        print(f"Error: not a directory: {root_dir}", file=sys.stderr)
        sys.exit(1)

    stats = rename_subfolders(root_dir, dry_run=args.dry_run)
    print(
        f"\nDone. Planned/Renamed: {stats['renamed']}, "
        f"Skipped (no match): {stats['skipped_no_match']}, "
        f"Skipped (target exists): {stats['skipped_exists']}, "
        f"Errors: {stats['errors']}"
    )


if __name__ == "__main__":
    main()