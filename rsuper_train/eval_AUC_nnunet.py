#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make tumour‑volume CSVs from NN‑U‑Net probability maps (*.npz).

   probabilities[18, D, H, W]
        ch 2  kidney  tumour
        ch 7  pancreatic tumour
        ch 8  liver    tumour
"""

import os, argparse, csv, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from filelock import FileLock

THR_LIST = [i / 10 for i in range(1, 10)]          # 0.1 … 0.9
TUMOUR_CH = {"liver": 8, "pancreatic": 7, "kidney": 2}

# ------------------------------------------------------------------ #
def write_csv_row(row: dict, cols: list[str], path: str):
    """append row under exclusive lock (columns sorted alphabetically)"""
    lock = FileLock(path + ".lock", timeout=30)
    with lock:
        header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if header:
                w.writeheader()
            w.writerow(row)

# ------------------------------------------------------------------ #
def volumes_from_npz(npz_path: str):
    data = np.load(npz_path)["probabilities"]       # (18, D, H, W)
    base_id = os.path.splitext(os.path.basename(npz_path))[0]  # BDMAP_XXXX

    # Prepare:  {thr: {col: value}}
    out = {thr: {"BDMAP_ID": base_id} for thr in THR_LIST}

    for organ, ch in TUMOUR_CH.items():
        prob = data[ch]                            # tumour prob map
        max_p = float(prob.max())

        for thr in THR_LIST:
            vol = int((prob >= thr).sum())
            col_vol  = f"{organ} tumor volume predicted"
            col_prob = f"{organ} tumor maximum probability"
            out[thr][col_vol]  = vol
            out[thr][col_prob] = max_p

    return out                                     # dict keyed by thr

# ------------------------------------------------------------------ #
def process_single(npz_file, csv_cols, csv_root):
    """wrapper for pool, writes 9 separate CSVs (one per threshold)"""
    try:
        rows_by_thr = volumes_from_npz(npz_file)
        for thr, row in rows_by_thr.items():
            write_csv_row(row, csv_cols,
                          csv_root.replace(".csv", f"_th{thr}.csv"))
    except Exception as e:
        print(f"[ERROR] {npz_file}: {e}")

# ------------------------------------------------------------------ #
def split_parts(lst, n, idx):
    base, extra = divmod(len(lst), n)
    s = idx * base + min(idx, extra)
    e = s + base + (1 if idx < extra else 0)
    return lst[s:] if idx == n-1 else lst[s:e]

# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_folder", required=True,
                    help="folder with BDMAP_*.npz")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--parts",   type=int, default=1)
    ap.add_argument("--part",    type=int, default=0)
    ap.add_argument("--cases",   help="CSV with BDMAP_ID column to subset")
    ap.add_argument("--continuing", action="store_true",
                    help="skip IDs already in *_th0.1.csv")
    args = ap.parse_args()

    npz_files = sorted(f for f in os.listdir(args.outputs_folder)
                       if f.endswith(".npz"))
    if args.cases:
        ids_keep = set(pd.read_csv(args.cases)["BDMAP_ID"])
        npz_files = [f for f in npz_files if f[:-4] in ids_keep]

    csv_root = os.path.join(args.outputs_folder, "tumor_detection_results.csv")

    # skip processed?
    if args.continuing and os.path.exists(csv_root.replace(".csv", "_th0.1.csv")):
        done = set(pd.read_csv(csv_root.replace(".csv", "_th0.1.csv"))["BDMAP_ID"])
        npz_files = [f for f in npz_files if f[:-4] not in done]

    if args.parts > 1:
        npz_files = split_parts(npz_files, args.parts, args.part)

    if not npz_files:
        print("Nothing to do.")
        return

    # establish column list once
    example = volumes_from_npz(os.path.join(args.outputs_folder, npz_files[0]))
    CSV_COLS = sorted(example[0.1].keys())

    pool = ProcessPoolExecutor(max_workers=args.workers)
    futures = [pool.submit(process_single,
                           os.path.join(args.outputs_folder, f),
                           CSV_COLS, csv_root) for f in npz_files]

    for _ in as_completed(futures):
        pass
    pool.shutdown()
    print("Finished.")

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()