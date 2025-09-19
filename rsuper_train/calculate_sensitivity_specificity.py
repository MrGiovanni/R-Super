import csv
import re
import argparse
import pandas as pd

def evaluate_predictions(ground_truth_csv, predictions_csv, output_csv):
    """
    Compare ground truth (report-based) with predicted volumes, computing
    sensitivity, specificity, and F1-Score at multiple volume thresholds.
    """

    # --------------------------
    # 0) Small helper: normalize IDs
    # --------------------------
    def normalize_id(val: str) -> str:
        """
        Make IDs merge-friendly:
          - cast to str
          - strip whitespace
          - remove a single trailing '.npz' if present
        """
        s = str(val).strip()
        if s.endswith(".npz"):
            s = s[:-4]
        return s

    # --------------------------
    # 1) Read ground truth CSV
    # --------------------------
    gt_df = pd.read_csv(ground_truth_csv)

    # Rename column "BDMAP ID" -> "BDMAP_ID" for consistency
    if "BDMAP ID" in gt_df.columns:
        gt_df = gt_df.rename(columns={"BDMAP ID": "BDMAP_ID"})

    # Normalize IDs (handles any accidental whitespace)
    gt_df["BDMAP_ID"] = gt_df["BDMAP_ID"].apply(normalize_id)

    # Binary ground truth: 1 if #instances >= 1, else 0
    gt_df["gt_liver"] = gt_df["number of liver lesion instances"].apply(lambda x: 1 if x >= 1 else 0)
    gt_df["gt_pancreatic"] = gt_df["number of pancreatic lesion instances"].apply(lambda x: 1 if x >= 1 else 0)
    gt_df["gt_kidney"] = gt_df["number of kidney lesion instances"].apply(lambda x: 1 if x >= 1 else 0)

    # De-duplicate if any duplicates exist after normalization
    if gt_df["BDMAP_ID"].duplicated().any():
        gt_df = gt_df.drop_duplicates(subset=["BDMAP_ID"], keep="last")

    # -------------------------
    # 2) Read predictions CSV
    # -------------------------
    pred_df = pd.read_csv(predictions_csv)

    # Normalize IDs to drop trailing ".npz" if present
    pred_df["BDMAP_ID"] = pred_df["BDMAP_ID"].apply(normalize_id)

    # Keep relevant columns
    relevant_cols = [
        "BDMAP_ID",
        "liver tumor volume predicted",
        "pancreatic tumor volume predicted",
        "kidney tumor volume predicted"
    ]
    pred_df = pred_df[relevant_cols]

    # De-duplicate in predictions if needed after normalization
    if pred_df["BDMAP_ID"].duplicated().any():
        pred_df = pred_df.drop_duplicates(subset=["BDMAP_ID"], keep="last")

    # -------------------------
    # 3) Merge on BDMAP_ID
    # -------------------------
    df_merged = pd.merge(gt_df, pred_df, on="BDMAP_ID", how="inner")

    # Organ mappings
    organ_gt_map = {
        "liver": "gt_liver",
        "pancreatic": "gt_pancreatic",
        "kidney": "gt_kidney"
    }

    organ_pred_map = {
        "liver": "liver tumor volume predicted",
        "pancreatic": "pancreatic tumor volume predicted",
        "kidney": "kidney tumor volume predicted"
    }

    organs = ["liver", "pancreatic", "kidney"]

    # -------------------------
    # 4) Thresholds (mm3). Use this when saving binary predictions (faster). If you save probabilities, you can use a confidence threshold.
    # -------------------------
    thresholds =  ([i * 10   for i in range(1, 10)]   +         #   10 …   90
                   [i * 10   for i in range(10, 100)] +         #  100 …  990
                   [i * 100  for i in range(1, 100)] +          #  100 … 9900
                   [i * 1000 for i in range(1, 100)])           # 1000 … 9 9000

    # -------------------------
    # 5) Helpers for formatting
    # -------------------------
    def format_metric(numer, denom):
        """
        Return "XX% (x/y)" or "N/A (0/0)" if denom=0.
        """
        if denom == 0:
            return "N/A (0/0)"
        perc = 100.0 * numer / denom
        return f"{perc:.1f}% ({numer}/{denom})"

    def format_f1(tp, fp, fn):
        """
        Return "XX% (TP=tp, FP=fp, FN=fn)" or "N/A (TP=0, FP=0, FN=0)" if denom=0.
        F1 = 2TP / (2TP + FP + FN)
        """
        denom = (2 * tp) + fp + fn
        if denom == 0:
            return "N/A (TP=0, FP=0, FN=0)"
        f1 = 100.0 * (2 * tp) / denom
        return f"{f1:.1f}% (TP={tp}, FP={fp}, FN={fn})"

    # We'll store final rows, one per threshold
    results = []

    # For each threshold, compute sensitivity/specificity/F1 for each organ
    for T in thresholds:
        row_data = {"threshold": T}
        for organ in organs:
            # Confusion matrix
            TP, FP, TN, FN = 0, 0, 0, 0

            # Go through all cases
            for _, row in df_merged.iterrows():
                gt_label = row[organ_gt_map[organ]]  # 0 or 1
                pred_volume = row[organ_pred_map[organ]]
                pred_label = 1 if pred_volume >= T else 0

                if gt_label == 1 and pred_label == 1:
                    TP += 1
                elif gt_label == 1 and pred_label == 0:
                    FN += 1
                elif gt_label == 0 and pred_label == 1:
                    FP += 1
                elif gt_label == 0 and pred_label == 0:
                    TN += 1

            # Sensitivity (Recall)
            sens_str = format_metric(TP, TP + FN)
            # Specificity
            spec_str = format_metric(TN, TN + FP)
            # F1-Score
            f1_str = format_f1(TP, FP, FN)

            row_data[f"{organ}_sensitivity"] = sens_str
            row_data[f"{organ}_specificity"] = spec_str
            row_data[f"{organ}_f1"] = f1_str

        results.append(row_data)

    # -------------------------
    # 6) Write output
    # -------------------------
    fieldnames = [
        "threshold",
        "liver_sensitivity", "liver_specificity", "liver_f1",
        "pancreatic_sensitivity", "pancreatic_specificity", "pancreatic_f1",
        "kidney_sensitivity", "kidney_specificity", "kidney_f1"
    ]

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_data in results:
            writer.writerow(row_data)

    print(f"Evaluation complete. Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth at multiple volume thresholds.")
    parser.add_argument("--ground_truth_csv", type=str, required=True,
                        help="Path to the ground truth CSV (report-based).")
    parser.add_argument("--predictions_csv", type=str, required=True,
                        help="Path to the predictions CSV (volumes).")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to output CSV where evaluation metrics will be saved.")
    args = parser.parse_args()

    evaluate_predictions(
        ground_truth_csv=args.ground_truth_csv,
        predictions_csv=args.predictions_csv,
        output_csv=args.output_csv
    )