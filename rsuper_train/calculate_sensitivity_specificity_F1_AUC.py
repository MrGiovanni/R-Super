#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create one metrics_thX.csv per confidence threshold X (0.1 … 0.9).

Each row (one tumour‑volume threshold) now contains:

threshold,
{organ}_sens   e.g. “86.8% (33/38)”
{organ}_spec   e.g. “92.5% (148/160)”
{organ}_f1     e.g. 0.874
{organ}_auc_prob   voxel‑probability AUROC (same value in every row)
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# -------------------------------------------------------------------- #
# Threshold grids
# -------------------------------------------------------------------- #
THRESHOLDS_CONF = [i / 10 for i in range(1, 10)]                # 0.1 … 0.9
THRESHOLDS_VOL  = ([i * 10   for i in range(1, 10)]   +         #   10 …   90
                   [i * 10   for i in range(10, 100)] +         #  100 …  990
                   [i * 100  for i in range(1, 100)] +          #  100 … 9900
                   [i * 1000 for i in range(1, 100)])           # 1000 … 9 9000

ORGANS = ["liver", "pancreatic", "kidney"]

# -------------------------------------------------------------------- #
# Pretty percentage strings
# -------------------------------------------------------------------- #
def pct_string(num, den):
    return "N/A (0/0)" if den == 0 else f"{100.0 * num / den:.1f}% ({num}/{den})"

def _f1(tp, fp, fn):
    d = 2*tp + fp + fn
    return (2*tp) / d if d else np.nan

# -------------------------------------------------------------------- #
# Load ground‑truth and create binary labels
# -------------------------------------------------------------------- #
def load_ground_truth(path):
    gt = pd.read_csv(path)

    if "BDMAP ID" in gt.columns:
        gt = gt.rename(columns={"BDMAP ID": "BDMAP_ID"})
    if "FELIX ID" in gt.columns:
        gt["gt_liver"]      = 0
        gt["gt_pancreatic"] = gt["FELIX ID"].str.contains("PDAC").astype(int)
        gt["gt_kidney"]     = 0
    elif "Diagnosis" in gt.columns:
        gt["gt_liver"]      = 0
        gt["gt_pancreatic"] = gt["Diagnosis"].str.contains(
            r"PDAC|lesion|PNET|cyst", regex=True
        ).astype(int)
        gt["gt_kidney"]     = 0
    else:  # numeric lesion counts
        gt["gt_liver"]      = (gt["number of liver lesion instances"]      >= 1).astype(int)
        gt["gt_pancreatic"] = (gt["number of pancreatic lesion instances"] >= 1).astype(int)
        gt["gt_kidney"]     = (gt["number of kidney lesion instances"]     >= 1).astype(int)
    return gt[["BDMAP_ID", "gt_liver", "gt_pancreatic", "gt_kidney"]]

# -------------------------------------------------------------------- #
# Voxel‑probability AUROC (same scalar per organ for all output files)
# -------------------------------------------------------------------- #
def prob_auc(gt, preds_df0):
    """
    Compute voxel‑probability AUROC for liver, pancreatic and kidney
    tumours.  NaNs in either GT or prediction column are removed *row by
    row* so the pairing between GT and predictions is preserved.

    Returns a dict {organ: auc_float_or_nan}.
    """
    # Merge once on ID
    merged = pd.merge(
        gt,
        preds_df0[[
            "BDMAP_ID",
            "liver tumor maximum probability",
            "pancreatic tumor maximum probability",
            "kidney tumor maximum probability"]],
        on="BDMAP_ID", how="inner"
    )

    out = {}

    for org in ORGANS:
        y_true = merged[f"gt_{org}"]
        y_prob = merged[f"{org} tumor maximum probability"]

        # same mask for both columns — keeps alignment
        mask = ~(y_true.isna() | y_prob.isna())
        y_true_clean = y_true[mask].astype(float)
        y_prob_clean = y_prob[mask].astype(float)

        # Need at least one positive & one negative sample
        if y_true_clean.nunique() < 2:
            out[org] = np.nan
            continue

        try:
            auc = roc_auc_score(y_true_clean, y_prob_clean)
            out[org] = round(auc, 3)
        except ValueError:
            # Covers constant predictions, etc.
            out[org] = np.nan

    return out



# -------------------------------------------------------------------- #
def evaluate_all(ground_truth_csv, preds_dir):
    gt = load_ground_truth(ground_truth_csv)

    # read all *_thX.csv
    pred_paths = sorted(glob.glob(os.path.join(preds_dir, "*results_th*.csv")))
    conf_from_name = lambda p: float(re.search(r"_th([0-9.]+)\.csv$", p).group(1))
    preds_all = {conf_from_name(p): pd.read_csv(p).drop_duplicates(subset='BDMAP_ID') for p in pred_paths}

    if set(preds_all) != set(THRESHOLDS_CONF):
        raise RuntimeError("Missing some *_thX.csv files")

    # scalar AUROC from maximum probabilities
    auc_prob_scalar = prob_auc(gt, preds_all[THRESHOLDS_CONF[0]])

    # ---------------------------------------------------------------- #
    # For each confidence threshold write its metrics CSV
    # ---------------------------------------------------------------- #
    for conf, pred_df in preds_all.items():
        df_merge = pd.merge(gt, pred_df, on="BDMAP_ID", how="inner")


        rows = []
        for vthr in THRESHOLDS_VOL:
            row = {"threshold": vthr}
            for organ in ORGANS:
                gt_bin  = df_merge[f"gt_{organ}"].values.astype(bool)
                vols    = df_merge[f"{organ} tumor volume predicted"].values
                preds   = (vols >= vthr)

                tp = np.logical_and(gt_bin,  preds).sum()
                fn = np.logical_and(gt_bin, ~preds).sum()
                fp = np.logical_and(~gt_bin, preds).sum()
                tn = np.logical_and(~gt_bin, ~preds).sum()

                row[f"{organ}_sens"] = pct_string(tp, tp + fn)
                row[f"{organ}_spec"] = pct_string(tn, tn + fp)
                row[f"{organ}_f1"]   = ("" if np.isnan(_f1(tp, fp, fn))
                                          else round(_f1(tp, fp, fn), 3))
                row[f"{organ}_auc_prob"] = auc_prob_scalar[organ]

            rows.append(row)

        out_csv = os.path.join(preds_dir, f"metrics_th{conf}.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("→ saved", out_csv)

# -------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground_truth_csv", required=True,
                    help="CSV with ground‑truth lesion counts / labels")
    ap.add_argument("--preds_dir", required=True,
                    help="Folder with *_th0.1.csv … *_th0.9.csv")
    args = ap.parse_args()

    evaluate_all(ground_truth_csv=args.ground_truth_csv,
                 preds_dir=args.preds_dir)