# ======================================================
# REPARSE GNINA RESULTS
# Reads existing *_out.pdbqt files and rebuilds CSV
# ======================================================

import os
import pandas as pd


def parse_score(pdbqt_file):
    """
    GNINA output parsing
    Priority:
      1) REMARK CNNaffinity
      2) REMARK CNNscore
      3) VINA RESULT
    """
    vina_score = None
    cnn_score = None

    with open(pdbqt_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("REMARK CNNaffinity"):
                try:
                    vina_score = float(line.split()[-1])
                    break
                except:
                    pass

            if line.startswith("REMARK CNNscore"):
                try:
                    cnn_score = float(line.split()[-1])
                except:
                    pass

            if "VINA RESULT" in line:
                try:
                    vina_score = float(line.split()[3])
                    break
                except:
                    pass

    if vina_score is None and cnn_score is not None:
        vina_score = cnn_score

    return vina_score


def main(result_dir, library_csv, output_csv):

    ligand_dir = os.path.join(result_dir, "ligands")

    # Load the original library
    df = pd.read_csv(library_csv)
    df["vina_score"] = None
    df["best_pose"] = None

    out_files = [
        f for f in os.listdir(result_dir)
        if f.endswith("_out.pdbqt")
    ]

    print(f"Found docking result files: {len(out_files)}")

    for fname in out_files:
        name = fname.replace("_out.pdbqt", "")

        try:
            idx = int(name.replace("ID_", ""))
        except ValueError:
            continue

        pdbqt_path = os.path.join(result_dir, fname)
        score = parse_score(pdbqt_path)

        if score is not None:
            df.at[idx, "vina_score"] = score
            df.at[idx, "best_pose"] = pdbqt_path

    print("Parsed vina_score count:",
          df["vina_score"].notnull().sum())

    df.to_csv(output_csv, index=False)
    print("Saved:", output_csv)


if __name__ == "__main__":

    # Example usage:
    # python reparse_gnina_results.py result_dir library.csv Reparsed_Docking.csv

    import sys

    if len(sys.argv) != 4:
        print("Usage: python reparse_gnina_results.py result_dir library_csv output_csv")
        sys.exit(1)

    result_dir = sys.argv[1]
    library_csv = sys.argv[2]
    output_csv = sys.argv[3]

    main(result_dir, library_csv, output_csv)
