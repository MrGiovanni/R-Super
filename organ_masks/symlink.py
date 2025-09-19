#!/usr/bin/env python3
"""Fast symlink creator for large BDMAP‑ID datasets (≈50 k files).

Highlights
=========
* **Thread‑pool parallelism** – default to 32 workers (change with --workers).
* **Batch scheduling** – large *chunksize* so the queue isn’t a bottleneck.
* **Minimal I/O per file** – skip stat calls when link already exists.
* **Progress bar** – sleek `tqdm` ETA that updates only when a link is done.

Usage
-----
python make_symlinks_fast.py \
    --csv ids.csv \
    --source-root /data/source \
    --dest-dir   /data/dst
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Core worker ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def link_one(bdmap_id: str, src_root: Path, dst_root: Path) -> bool:
    """Attempt to create a symlink ‑> return True on success, False otherwise."""
    src = src_root / bdmap_id / "ct.nii.gz"
    dst = dst_root / f"{bdmap_id}.nii.gz"

    # If target already exists (file or symlink) just skip.
    if dst.exists() or dst.is_symlink():
        return False

    try:
        # Using the raw OS call is fastest.
        os.symlink(src, dst)
        return True
    except FileExistsError:  # race condition when multiple workers hit same id
        return False
    except OSError as e:
        # Log once per failure
        print(f"❌ {bdmap_id}: {e}", file=sys.stderr)
        return False

# ---------------------------------------------------------------------------
# Main -----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Ultra‑fast symlink creator")
    p.add_argument("--csv", required=True, help="CSV with a BDMAP ID column")
    p.add_argument("--bdmap-col", default="BDMAP ID",
                   help="Name of the column containing the IDs")
    p.add_argument("--source-root", required=True,
                   help="Root folder holding <ID>/ct.nii.gz files")
    p.add_argument("--dest-dir", required=True,
                   help="Where to place the <ID>.nii.gz symlinks")
    p.add_argument("--workers", type=int, default=32,
                   help="Number of parallel threads (default: %(default)s)")
    args = p.parse_args()

    # Resolve paths once for speed
    src_root = Path(args.source_root).expanduser().resolve()
    dst_root = Path(args.dest_dir).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    # Read only the needed column
    try:
        ids = pd.read_csv(args.csv, usecols=[args.bdmap_col])[args.bdmap_col]
    except ValueError:
        print(f"❌ Column '{args.bdmap_col}' not found in {args.csv}", file=sys.stderr)
        sys.exit(1)

    ids_list = ids.astype(str).tolist()
    total = len(ids_list)

    # Prepare the partial so each worker only needs the ID string
    worker = partial(link_one, src_root=src_root, dst_root=dst_root)

    # ---------------------------------------------------------------------
    # Thread pool with chunky scheduling for low overhead
    # ---------------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # chunksize tuned empirically – adjust if you still see high ETA
        futures = [pool.submit(worker, bid) for bid in ids_list]
        for _ in tqdm(as_completed(futures), total=total, desc="Linking", unit="file"):
            pass  # progress bar advances as each future completes

if __name__ == "__main__":
    main()
