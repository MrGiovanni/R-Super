import csv
import argparse
import pandas as pd

def evaluate_predictions(ground_truth_csv, predictions_csv, output_csv):
    """
    Compare ground truth (report-based) with predicted volumes, computing
    sensitivity and specificity at multiple volume thresholds.

    Ground truth CSV columns (relevant ones):
        - "BDMAP ID" (unique case identifier)
        - "number of liver lesion instances"
        - "number of pancreatic lesion instances"
        - "number of kidney lesion instances"

    Predictions CSV columns (from previous code):
        - "BDMAP_ID"  (unique case identifier, matching ground truth's "BDMAP ID")
        - "liver tumor volume predicted"
        - "pancreatic tumor volume predicted"
        - "kidney tumor volume predicted"

    We define ground_truth_label = 1 if #lesion_instances >=1, else 0.
    Then for each threshold T, predicted_label = 1 if volume_predicted >= T, else 0.

    We'll compute for each organ:
        - sensitivity = 100 * TP / (TP + FN)
        - specificity = 100 * TN / (TN + FP)

    Output CSV will have columns:
        threshold,
        liver_sensitivity, liver_specificity,
        pancreatic_sensitivity, pancreatic_specificity,
        kidney_sensitivity, kidney_specificity

    Each row corresponds to a single threshold.
    """

    # --------------------------
    # 1) Read ground truth CSV
    # --------------------------
    gt_df = pd.read_csv(ground_truth_csv)

    # Rename column "BDMAP ID" -> "BDMAP_ID" for consistency
    if "BDMAP ID" in gt_df.columns:
        gt_df = gt_df.rename(columns={"BDMAP ID": "BDMAP_ID"})

    # Binary ground truth: 1 if #instances >= 1, else 0
    gt_df["gt_liver"] = gt_df["number of liver lesion instances"].apply(lambda x: 1 if x >= 1 else 0)
    gt_df["gt_pancreatic"] = gt_df["number of pancreatic lesion instances"].apply(lambda x: 1 if x >= 1 else 0)
    gt_df["gt_kidney"] = gt_df["number of kidney lesion instances"].apply(lambda x: 1 if x >= 1 else 0)

    # -------------------------
    # 2) Read predictions CSV
    # -------------------------
    pred_df = pd.read_csv(predictions_csv)
    # Keep relevant columns
    relevant_cols = [
        "BDMAP_ID",
        "liver tumor volume predicted",
        "pancreatic tumor volume predicted",
        "kidney tumor volume predicted"
    ]
    pred_df = pred_df[relevant_cols]

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
    # 4) Thresholds
    # -------------------------
    thresholds = [1, 10, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000]

    # -------------------------
    # 5) Helper for formatting
    # -------------------------
    def format_metric(numer, denom):
        """
        Return "XX% (x/y)" or "N/A (0/0)" if denom=0.
        """
        if denom == 0:
            return "N/A (0/0)"
        perc = 100.0 * numer / denom
        return f"{perc:.1f}% ({numer}/{denom})"

    # We'll store final rows, one per threshold
    results = []

    # For each threshold, compute sensitivity/specificity for each organ
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

            # Sensitivity
            sens_str = format_metric(TP, TP + FN)
            # Specificity
            spec_str = format_metric(TN, TN + FP)

            # Example: "liver_sensitivity" = "75% (3/4)"
            row_data[f"{organ}_sensitivity"] = sens_str
            row_data[f"{organ}_specificity"] = spec_str

        results.append(row_data)

    # -------------------------
    # 6) Write output
    # -------------------------
    # We'll have columns:
    # ["threshold",
    #  "liver_sensitivity", "liver_specificity",
    #  "pancreatic_sensitivity", "pancreatic_specificity",
    #  "kidney_sensitivity", "kidney_specificity"]
    fieldnames = [
        "threshold",
        "liver_sensitivity", "liver_specificity",
        "pancreatic_sensitivity", "pancreatic_specificity",
        "kidney_sensitivity", "kidney_specificity"
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