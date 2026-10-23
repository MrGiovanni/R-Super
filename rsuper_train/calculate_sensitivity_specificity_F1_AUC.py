#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create metrics from one predictions CSV, or one metrics_thX.csv per
confidence threshold X (0.1 … 0.9).

Each row (one tumour‑volume threshold) now contains:

threshold,
{organ}_sens   e.g. “86.8% (33/38)”
{organ}_spec   e.g. “92.5% (148/160)”
{organ}_f1     e.g. 0.874
{organ}_auc_prob or {organ}_auc_volume   case-level AUROC, using maximum
voxel probability or predicted lesion volume as the continuous score

Finally print each organ's best balanced accuracy, sensitivity, specificity,
volume threshold and AUROC.
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
# Case-level AUROC
# -------------------------------------------------------------------- #
def score_auc(gt, preds_df, score_kind):
    """
    Compute case-level AUROC from continuous probability or volume scores.
    NaNs are removed in matched GT/prediction pairs.

    Returns a dict {organ: auc_float_or_nan}.
    """
    score_suffix = {"prob": "maximum probability", "volume": "volume predicted"}[score_kind]
    merged = pd.merge(
        gt,
        preds_df[["BDMAP_ID"] + [f"{org} tumor {score_suffix}" for org in ORGANS]],
        on="BDMAP_ID", how="inner"
    )

    out = {}

    for org in ORGANS:
        y_true = merged[f"gt_{org}"]
        y_prob = merged[f"{org} tumor {score_suffix}"]

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
            # Invalid scores, e.g. infinite values.
            out[org] = np.nan

    return out


def prob_auc(gt, preds_df0):
    return score_auc(gt, preds_df0, "prob")


# -------------------------------------------------------------------- #
def evaluate_all(ground_truth_csv, preds_dir=None, predictions_csv=None,
                 output_csv=None, auc_score="prob"):
    gt = load_ground_truth(ground_truth_csv)

    if predictions_csv is not None:
        preds_all = {None: pd.read_csv(predictions_csv).drop_duplicates(subset="BDMAP_ID")}
    else:
        # read all *_thX.csv
        pred_paths = sorted(glob.glob(os.path.join(preds_dir, "*results_th*.csv")))
        conf_from_name = lambda p: float(re.search(r"_th([0-9.]+)\.csv$", p).group(1))
        preds_all = {conf_from_name(p): pd.read_csv(p).drop_duplicates(subset='BDMAP_ID') for p in pred_paths}

        if set(preds_all) != set(THRESHOLDS_CONF):
            raise RuntimeError("Missing some *_thX.csv files")

    # Preserve the shared probability AUROC; volumes vary with confidence.
    if auc_score == "prob":
        auc_prob_scalar = prob_auc(gt, next(iter(preds_all.values())))

    # ---------------------------------------------------------------- #
    # For each confidence threshold write its metrics CSV
    # ---------------------------------------------------------------- #
    for conf, pred_df in preds_all.items():
        df_merge = pd.merge(gt, pred_df, on="BDMAP_ID", how="inner")
        auc_scalar = (score_auc(gt, pred_df, "volume")
                      if auc_score == "volume" else auc_prob_scalar)

        rows = []
        best_balacc = {organ: None for organ in ORGANS}
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

                if tp + fn > 0 and tn + fp > 0:
                    sensitivity = tp / (tp + fn)
                    specificity = tn / (tn + fp)
                    balacc = (sensitivity + specificity) / 2
                    best = best_balacc[organ]
                    if best is None or (balacc, -vthr) > (best[0], -best[3]):
                        best_balacc[organ] = (balacc, sensitivity, specificity, vthr)

                row[f"{organ}_sens"] = pct_string(tp, tp + fn)
                row[f"{organ}_spec"] = pct_string(tn, tn + fp)
                row[f"{organ}_f1"]   = ("" if np.isnan(_f1(tp, fp, fn))
                                          else round(_f1(tp, fp, fn), 3))
                row[f"{organ}_auc_{auc_score}"] = auc_scalar[organ]

            rows.append(row)

        out_csv = (output_csv or os.path.join(os.path.dirname(predictions_csv), "metrics_auc.csv")
                   if predictions_csv is not None
                   else os.path.join(preds_dir, f"metrics_th{conf}.csv"))
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("→ saved", out_csv)
        print(f"Best balanced accuracy by organ ({auc_score} AUROC; lowest threshold breaks ties):")
        for organ in ORGANS:
            auc = auc_scalar[organ]
            auc_str = "N/A" if np.isnan(auc) else f"{auc:.3f}"
            best = best_balacc[organ]
            if best is None:
                print(f"{organ}: balacc=N/A, sen=N/A, spec=N/A, th=N/A, auc={auc_str} "
                      "(requires positive and negative cases)")
                continue
            balacc, sensitivity, specificity, threshold = best
            print(f"{organ}: balacc={balacc:.2%}, sen={sensitivity:.2%}, "
                  f"spec={specificity:.2%}, th={threshold} mm3, auc={auc_str}")

# -------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground_truth_csv", required=True,
                    help="CSV with ground‑truth lesion counts / labels")
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--preds_dir",
                    help="Folder with *_th0.1.csv … *_th0.9.csv")
    inputs.add_argument("--predictions_csv",
                        help="Single CSV with predicted lesion volumes / probabilities")
    ap.add_argument("--auc_score", choices=["prob", "volume"], default="prob",
                    help="Continuous score for case-level AUROC (default: prob)")
    ap.add_argument("--output_csv",
                    help="Output for --predictions_csv (default: metrics_auc.csv beside input)")
    args = ap.parse_args()
    if args.output_csv and not args.predictions_csv:
        ap.error("--output_csv requires --predictions_csv")

    evaluate_all(ground_truth_csv=args.ground_truth_csv,
                 preds_dir=args.preds_dir,
                 predictions_csv=args.predictions_csv,
                 output_csv=args.output_csv,
                 auc_score=args.auc_score)
