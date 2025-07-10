import pandas as pd
import argparse

from tqdm import tqdm  # Optional: progress bar in loops


def keep_last_answer(
        df: pd.DataFrame,
        id_col: str = "Encrypted Accession Number",
        answer_col: str = "DNN Answer"
) -> pd.DataFrame:
    """
    Return a copy of *df* where, for every BDMAP_ID that has more than one
    distinct LLM answer, only the rows belonging to the **last** distinct
    answer (by first appearance order) are kept.  
    If a given ID has just one answer, all its rows are kept.

    Parameters
    ----------
    df : pd.DataFrame
        The tumour-level dataframe (one row per tumour).
    id_col : str, default "BDMAP ID"
        Column that identifies the study / patient / sample.
    answer_col : str, default "DNN Answer"
        Column that stores the raw LLM output.

    Returns
    -------
    pd.DataFrame
        A *new* dataframe, row-order preserved, with the filtered content.
    """

    # we will collect the surviving row indices
    keep_idx = []

    for bid, grp in df.groupby(id_col, sort=False):  # keep original order
        # list unique answers in the order they appear in the CSV
        unique_answers = grp[answer_col].drop_duplicates().tolist()

        if len(unique_answers) <= 1:
            # nothing to deduplicate – keep everything
            keep_idx.extend(grp.index)
        else:
            # take the *last* distinct answer
            last_answer = unique_answers[-1]
            keep_idx.extend(grp[grp[answer_col] == last_answer].index)

    # return the filtered frame (preserve original row order)
    df = df.loc[keep_idx].copy()
    df = df.drop_duplicates(subset=[id_col, 'Tumor ID'], keep='last')
    df = df.dropna(subset=[id_col])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Filter a DataFrame to keep only the last distinct answer per ID."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to write the filtered CSV file."
    )
    

    args = parser.parse_args()

    # Read the data
    df = pd.read_csv(args.input)

    # Apply filtering
    result = keep_last_answer(
        df
    )

    # Save to CSV
    result.to_csv(args.output, index=False)
    print(f"Filtered data written to {args.output}")


if __name__ == "__main__":
    main()