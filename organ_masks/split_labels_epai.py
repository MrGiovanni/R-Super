#!/usr/bin/env python3
"""
nnUNet combined labels → one-hot BDMAP labels
• skips cases already converted
• supports N-way splitting across machines   (--parts / --part)
"""
import argparse
import math
import os
from multiprocessing import Pool, cpu_count

import nibabel as nib
import numpy as np
from tqdm import tqdm

# ───────────────────────── CLI ──────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input_nnunet_gt_path", required=True,
                    help="Folder with <BDMAP_ID>.nii.gz combined labels")
parser.add_argument("--output_bdmap_gt_path", required=True,
                    help="Destination root (one-hot masks will be placed here)")
parser.add_argument("--num_processes", type=int,
                    default=max(1, int(cpu_count() * .8)),
                    help="Worker processes per machine (default ≈ 80%% of CPUs)")
parser.add_argument("--parts", type=int, default=1,
                    help="Total slices of the job (for multi-machine use)")
parser.add_argument("--part", type=int, default=0,
                    help="Which slice to run on this machine (0 … parts-1)")
args = parser.parse_args()

if args.part < 0 or args.part >= args.parts:
    raise ValueError(f"--part must be in [0, {args.parts-1}]")

# ─────────────────────── label list ─────────────────────
LABEL_MAPPING = [
    ('aorta.nii.gz',                         1),
    ('adrenal_gland_left.nii.gz',            2),
    ('adrenal_gland_right.nii.gz',           3),
    ('common_bile_duct.nii.gz',              4),
    ('celiac_aa.nii.gz',                     5),
    ('colon.nii.gz',                         6),
    ('duodenum.nii.gz',                      7),
    ('gall_bladder.nii.gz',                  8),
    ('postcava.nii.gz',                      9),
    ('kidney_left.nii.gz',                  10),
    ('kidney_right.nii.gz',                 11),
    ('liver.nii.gz',                        12),
    ('pancreas.nii.gz',                     13),
    ('pancreatic_duct.nii.gz',              14),
    ('superior_mesenteric_artery.nii.gz',   15),
    ('intestine.nii.gz',                    16),
    ('spleen.nii.gz',                       17),
    ('stomach.nii.gz',                      18),
    ('veins.nii.gz',                        19),
    ('renal_vein_left.nii.gz',              20),
    ('renal_vein_right.nii.gz',             21),
    ('cbd_stent.nii.gz',                    22),
    ('pancreatic_pdac.nii.gz',              23),
    ('pancreatic_cyst.nii.gz',              24),
    ('pancreatic_pnet.nii.gz',              25),
]

# ────────────────── helpers ──────────────────
def outputs_exist(case_id: str) -> bool:
    """Return True iff *all* one-hot files for case are present."""
    dst_dir = os.path.join(args.output_bdmap_gt_path, case_id, "segmentations")
    return all(os.path.isfile(os.path.join(dst_dir, fn)) for fn, _ in LABEL_MAPPING)

# ───────────────── per-case worker ─────────────────
def process_case(label_fname: str) -> str | None:
    case_id  = label_fname[:-7]                          # strip ".nii.gz"
    if outputs_exist(case_id):
        return None                                      # already done

    src_path = os.path.join(args.input_nnunet_gt_path, label_fname)
    dst_dir  = os.path.join(args.output_bdmap_gt_path, case_id, "segmentations")
    os.makedirs(dst_dir, exist_ok=True)

    img   = nib.load(src_path)
    data  = img.get_fdata().astype(np.uint8)
    aff   = img.affine
    hdr   = img.header.copy()
    hdr.set_data_dtype(np.uint8)

    for fname, idx in LABEL_MAPPING:
        mask = (data == idx).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask, aff, hdr), os.path.join(dst_dir, fname))

    return f"processed {case_id}"

# ────────────────────────── main ─────────────────────────
if __name__ == "__main__":
    all_labels = sorted(f for f in os.listdir(args.input_nnunet_gt_path)
                        if f.endswith(".nii.gz"))

    # filter out cases already done (cheap check – no file I/O inside workers)
    todo = [f for f in all_labels if not outputs_exist(f[:-7])]

    if not todo:
        print("Nothing to do – every case already converted.")
        exit(0)

    # ---------- slice the todo list ----------
    # even partition: floor/ceil trick
    n_total = len(todo)
    start   = (n_total * args.part)   // args.parts
    end     = (n_total * (args.part+1)) // args.parts
    todo_slice = todo[start:end]

    print(f"Total pending cases : {n_total}")
    print(f"Running slice       : {args.part+1}/{args.parts} "
          f"({len(todo_slice)} cases)")

    # ---------- parallel conversion ----------
    with Pool(args.num_processes) as pool:
        for msg in tqdm(pool.imap_unordered(process_case, todo_slice),
                        total=len(todo_slice), desc="Cases"):
            if msg:                       # None ⇒ skipped
                tqdm.write(msg)

    print("Finished!")