# ======================================================
# SCORE ONLY ENGINE (APP-CALLABLE FINAL VERSION)
# ======================================================

import os
import argparse
import pandas as pd
from scoring import compute_composite


def ensure_columns(df):
    if "vina_score" not in df.columns:
        df["vina_score"] = None
    if "cnn_score" not in df.columns:
        df["cnn_score"] = 0
    if "tox_score" not in df.columns:
        df["tox_score"] = 0
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Docking_Reparsed.csv")
    parser.add_argument("output_csv", help="Scored output CSV")
    parser.add_argument("--top_pct", type=float, default=0.05,
                        help="Top percentage for hit selection (default=0.05)")
    parser.add_argument("--top_n", type=int, default=None,
                        help="Top N for hit selection (optional)")
    parser.add_argument("--vina_only", action="store_true",
                        help="Rank by vina_score only (lower is better)")
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(args.input_csv)

    df = pd.read_csv(args.input_csv)
    df = ensure_columns(df)

    # Keep only successfully docked entries
    df = df[df["vina_score"].notnull()].copy()
    if len(df) == 0:
        raise RuntimeError("No valid vina_score rows found.")

    # Compute composite score
    df = compute_composite(df)
    df["composite_score"] = pd.to_numeric(df["composite_score"], errors="coerce")

    # Save results
    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    # Ranking criteria
    if args.vina_only:
        df_rank = df.sort_values("vina_score", ascending=True)
        rank_key = "vina_score"
    else:
        df_rank = df.sort_values("composite_score", ascending=False)
        rank_key = "composite_score"

    # Decide number of top hits
    if args.top_n is not None:
        k = max(1, args.top_n)
    else:
        k = max(1, int(len(df_rank) * args.top_pct))

    df_top = df_rank.head(k)

    # top_hits directory
    top_dir = os.path.join(out_dir, "top_hits")
    os.makedirs(top_dir, exist_ok=True)

    top_csv = os.path.join(top_dir, f"TopHits_{rank_key}_{k}.csv")
    df_top.to_csv(top_csv, index=False)

    print(f"[score_only] Scored: {len(df)}")
    print(f"[score_only] Top hits: {k}")
    print(f"[score_only] Saved: {top_csv}")


if __name__ == "__main__":
    main()
