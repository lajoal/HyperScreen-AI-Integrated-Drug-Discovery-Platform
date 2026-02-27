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


def check_stop_signal(result_dir: str) -> bool:
    """Check whether a stop signal file (stop.flag) exists in the result directory."""
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

    # ======================================================
    # Step 1) Load and filter the library (organic drug-like only)
    # ======================================================
    df = pd.read_csv(library_csv)

    if "SMILES" not in df.columns:
        print("❌ 'SMILES' column not found in the library CSV.")
        sys.exit(0)

    def _guess_name_col(_df: pd.DataFrame) -> str:
        """Guess a reasonable compound-name column if present."""
        candidates = [
            "Compound_Name", "COMPOUND_NAME", "compound_name",
            "Compound", "COMPOUND", "Name", "NAME", "Drug", "DRUG",
        ]
        for c in candidates:
            if c in _df.columns:
                return c
        return ""

    name_col = _guess_name_col(df)

    def is_valid(smiles, name=None) -> bool:
        """
        Keep organic, drug-like small molecules only.
        Exclude inorganic salts, metals, and radiopharmaceuticals (e.g., Ra-223 / Xofigo).
        """
        try:
            if smiles is None:
                return False

            smi = str(smiles).strip()
            if not smi:
                return False

            nm = (str(name).upper().strip() if name is not None else "")

            # 1) Name-based hard exclusions (radiopharmaceuticals / radionuclides)
            bad_keywords = [
                "RADIUM", "RA-223", "RA 223", "XOFIGO",
                "RADIO", "RADIONUCL", "RADIONUCLIDE",
                "I-131", "I 131", "IODINE-131", "IODINE 131",
                "Y-90", "Y 90", "LU-177", "LU 177",
                "TC-99", "TC 99", "TECHNETIUM-99", "TECHNETIUM 99",
            ]
            if nm and any(k in nm for k in bad_keywords):
                return False

            m = Chem.MolFromSmiles(smi)
            if m is None:
                return False

            # 2) Must contain carbon (organic-like)
            if not any(a.GetAtomicNum() == 6 for a in m.GetAtoms()):
                return False

            # 3) Exclude metals / non-organic atoms using an allowed atom set
            # Common organic atoms: H, C, N, O, F, P, S, Cl, Br, I
            allowed = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
            for a in m.GetAtoms():
                if a.GetAtomicNum() not in allowed:
                    return False

            # 4) Exclude tiny fragments (prevents small ions/fragments)
            if m.GetNumHeavyAtoms() < 8:
                return False

            # 5) Keep the MW criterion (>= 200)
            return Descriptors.MolWt(m) >= 200

        except Exception:
            return False

    print("Step 1: Filtering library (organic drug-like only, MW >= 200)...")
    if name_col:
        df = df[df.apply(lambda r: is_valid(r.get("SMILES"), r.get(name_col)), axis=1)].reset_index(drop=True)
    else:
        df = df[df["SMILES"].apply(lambda s: is_valid(s, None))].reset_index(drop=True)

    if len(df) == 0:
        print("❌ No valid molecules found after filtering.")
        sys.exit(0)

    # ======================================================
    # Step 2) PREP
    # ======================================================
    if check_stop_signal(result_dir):
        sys.exit(0)

    print("Step 2: Starting PREP...")
    df = run_prep_parallel(df, os.path.join(result_dir, "ligands"))

    # ======================================================
    # Step 3) DOCKING
    # ======================================================
    if check_stop_signal(result_dir):
        sys.exit(0)

    print("Step 3: Starting DOCKING (GNINA)...")
    df = run_docking_parallel(df, receptor, cx, cy, cz, result_dir)

    # ======================================================
    # Step 4) SCORING
    # ======================================================
    print("Step 4: Computing composite scores...")
    df = compute_composite(df)

    # ======================================================
    # Step 5) Post-optimization (top 10 only)
    # ======================================================
    # opt_rmsd: internal raw RMSD (fallback sentinel = 0.0001)
    # md_score: normalized score (not computed = 0.0)
    # opt_rmsd_display: display-only value (not computed = NA)
    df["opt_rmsd"] = 0.0001
    df["md_score"] = 0.0
    df["opt_rmsd_display"] = pd.NA

    top_n = min(len(df), 10)
    df_md = df.nlargest(top_n, "composite_score")

    md_dir = os.path.join(result_dir, "md")
    os.makedirs(md_dir, exist_ok=True)

    print(f"Step 5: Running ligand geometry optimization for top {top_n} candidates...")
    for idx, row in df_md.iterrows():
        if check_stop_signal(result_dir):
            break

        pose = row.get("best_pose")
        smiles = row.get("SMILES")

        if pose and os.path.exists(pose):
            val = run_md(pose, md_dir, receptor, smiles=smiles)

            # Guard against fallback/sentinel values
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

                md_score = float(np.exp(-val / 2.0))
                df.at[idx, "md_score"] = md_score
                df.at[idx, "opt_rmsd_display"] = round(val, 4)

                print(
                    f"   > Optimization completed for {row.get('Compound_Name', idx)}: "
                    f"RMSD={val:.4f}, md_score={md_score:.4f}"
                )
        else:
            df.at[idx, "opt_rmsd"] = 0.0001
            df.at[idx, "md_score"] = 0.0
            df.at[idx, "opt_rmsd_display"] = pd.NA
            print(f"   > Optimization skipped for {row.get('Compound_Name', idx)}: pose not found")

        # Incremental save (keeps the original behavior)
        df.to_csv(os.path.join(result_dir, "Final_AutoPipeline.csv"), index=False)

    # Safety pass for display column
    df["opt_rmsd_display"] = df["opt_rmsd_display"].astype("object")
    df.loc[pd.to_numeric(df["md_score"], errors="coerce") <= 0, "opt_rmsd_display"] = pd.NA

    mask_fill = (pd.to_numeric(df["md_score"], errors="coerce") > 0) & (df["opt_rmsd_display"].isna())
    df.loc[mask_fill, "opt_rmsd_display"] = pd.to_numeric(df.loc[mask_fill, "opt_rmsd"], errors="coerce").round(4)

    # ======================================================
    # Step 6) Final save (master output for all candidates)
    # ======================================================
    final_path = os.path.join(result_dir, "Final_AutoPipeline.csv")
    df.to_csv(final_path, index=False)

    # ======================================================
    # Step 7) screening_report (top-10 summary)
    # ======================================================
    try:
        report_cols = [
            "Compound_Name", "SMILES",
            "vina_score", "cnn_score", "toxicity_score",
            "vina_norm", "cnn_norm", "tox_norm",
            "composite_score",
            "opt_rmsd_display", "md_score",
            "best_pose",
        ]

        df_report = df.sort_values("composite_score", ascending=False).head(10).copy()
        report_cols = [c for c in report_cols if c in df_report.columns]

        report_path = os.path.join(result_dir, "screening_report.csv")
        df_report[report_cols].to_csv(report_path, index=False)

        print(f"✅ Screening report saved (top 10): {report_path}")
    except Exception as e:
        print(f"⚠️ Failed to save screening_report: {e}")

    # Completion flag
    with open(os.path.join(result_dir, "run_completed.flag"), "w") as f:
        f.write("done")

    print(f"✅ Pipeline finished successfully: {final_path}")


if __name__ == "__main__":
    main()
