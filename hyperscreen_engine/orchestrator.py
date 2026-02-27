import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from docking_parallel import run_docking_parallel
from scoring import compute_composite
from md_runner import run_md
from prep_parallel import run_prep_parallel
from config import *


def check_stop_signal(result_dir):
    """중단 신호(stop.flag) 파일이 있는지 확인"""
    return os.path.exists(os.path.join(result_dir, "stop.flag"))


def main():
    if len(sys.argv) != 7:
        sys.exit(1)

    receptor = sys.argv[1]
    cx = float(sys.argv[2])
    cy = float(sys.argv[3])
    cz = float(sys.argv[4])
    library_csv = sys.argv[5]
    result_dir = sys.argv[6]

    os.makedirs(result_dir, exist_ok=True)

    # 1) Load and filter the library (MW >= 200)
    df = pd.read_csv(library_csv)

    def is_valid(s):
        try:
            m = Chem.MolFromSmiles(str(s))
            return Descriptors.MolWt(m) >= 200 if m else False
        except Exception:
            return False

    print("Step 1: Filtering library (MW >= 200)...")
    df = df[df["SMILES"].apply(is_valid)].reset_index(drop=True)

    if len(df) == 0:
        print("❌ No valid molecules found after filtering.")
        sys.exit(0)

    # 2. PREP
    if check_stop_signal(result_dir):
        sys.exit(0)

    print("Step 2: Starting PREP...")
    df = run_prep_parallel(df, os.path.join(result_dir, "ligands"))

    # 3. DOCKING
    if check_stop_signal(result_dir):
        sys.exit(0)

    print("Step 3: Starting DOCKING (GNINA)...")
    df = run_docking_parallel(df, receptor, cx, cy, cz, result_dir)

    # 4. SCORING
    print("Step 4: Computing Composite Scores...")
    df = compute_composite(df)

    # 5. MD surrogate (top 10 only)
    # - opt_rmsd: raw RMSD stored for internal use (fallback = 0.0001)
    # - md_score : normalized score (not computed = 0.0)
    # - opt_rmsd_display: display-only value (not computed = NA)
    df["opt_rmsd"] = 0.0001
    df["md_score"] = 0.0
    df["opt_rmsd_display"] = pd.NA  # Display-only column (initially NA)

    top_n = min(len(df), 10)
    df_md = df.nlargest(top_n, "composite_score")

    md_dir = os.path.join(result_dir, "md")
    os.makedirs(md_dir, exist_ok=True)

    print(f"Step 5: Starting Ligand Geometry Optimization for top {top_n} candidates...")
    for idx, row in df_md.iterrows():
        if check_stop_signal(result_dir):
            break

        pose = row.get("best_pose")
        smiles = row.get("SMILES")

        if pose and os.path.exists(pose):
            val = run_md(pose, md_dir, receptor, smiles=smiles)

            # Guard against run_md fallback/sentinel values
            # Treat 0.0001 / 0.001 as failure/abnormal sentinels and set md_score = 0
            if val is None or float(val) <= 0.0011:
                df.at[idx, "opt_rmsd"] = 0.0001
                df.at[idx, "md_score"] = 0.0
                df.at[idx, "opt_rmsd_display"] = pd.NA
                print(
                    f"   > Optimization fallback for {row.get('Compound_Name', idx)}: "
                    f"RMSD sentinel={val}, md_score=0.0"
                )
            else:
                val = float(val)
                df.at[idx, "opt_rmsd"] = val

                # md_score: higher when RMSD is smaller (roughly 0..1)
                md_score = float(np.exp(-val / 2.0))
                df.at[idx, "md_score"] = md_score

                # For display, only keep values that were actually computed
                df.at[idx, "opt_rmsd_display"] = round(val, 4)

                print(
                    f"   > Optimization completed for {row.get('Compound_Name', idx)}: "
                    f"RMSD = {val:.4f}, md_score = {md_score:.4f}"
                )
        else:
            # Keep fallback (not computed)
            df.at[idx, "opt_rmsd"] = 0.0001
            df.at[idx, "md_score"] = 0.0
            df.at[idx, "opt_rmsd_display"] = pd.NA
            print(f"   > Optimization skipped for {row.get('Compound_Name', idx)}: pose not found")

        # Keep incremental saving (preserve existing behavior)
        # The display column is saved together
        df.to_csv(os.path.join(result_dir, "Final_AutoPipeline.csv"), index=False)

    # Safely post-process display column (in case any rows were missed)
    # If md_score <= 0, set the display value to NA
    df["opt_rmsd_display"] = df["opt_rmsd_display"].astype("object")
    df.loc[df["md_score"] <= 0, "opt_rmsd_display"] = pd.NA

    # If the value was computed but opt_rmsd_display is empty, fill it with opt_rmsd
    mask_fill = (df["md_score"] > 0) & (df["opt_rmsd_display"].isna())
    df.loc[mask_fill, "opt_rmsd_display"] = df.loc[mask_fill, "opt_rmsd"].round(4)

    # 6) Final save (master results: all candidates)
    final_path = os.path.join(result_dir, "Final_AutoPipeline.csv")
    df.to_csv(final_path, index=False)

    # 7) Save screening_report (top-10 summary based on Final_AutoPipeline)
    try:
        report_cols = [
            "Compound_Name", "SMILES",
            "vina_score", "cnn_score", "toxicity_score",
            "vina_norm", "cnn_norm", "tox_norm",
            "composite_score",
            "opt_rmsd_display", "md_score",
            "best_pose"
        ]

        df_report = (
            df.sort_values("composite_score", ascending=False)
              .head(10)
              .copy()
        )

        # Avoid column-name discrepancies across environments
        report_cols = [c for c in report_cols if c in df_report.columns]

        report_path = os.path.join(result_dir, "screening_report.csv")
        df_report[report_cols].to_csv(report_path, index=False)

        print(f"✅ Screening report saved (top 10): {report_path}")
    except Exception as e:
        print(f"⚠️ screening_report save failed: {e}")

    # Completion flag
    with open(os.path.join(result_dir, "run_completed.flag"), "w") as f:
        f.write("done")

    print(f"✅ Pipeline finished successfully: {final_path}")


if __name__ == "__main__":
    main()