import os
import sys
import json
import subprocess
import pandas as pd

GNINA_BIN = "gnina"  # Use as-is if available in PATH

def main(result_dir: str):
    result_dir = os.path.abspath(result_dir)
    print(f"[REFINE] Result dir: {result_dir}")

    # --------------------------------------------------
    # 1) Load meta.json
    # --------------------------------------------------
    meta_path = os.path.join(result_dir, "meta.json")
    if not os.path.exists(meta_path):
        print("[REFINE][ERROR] meta.json not found")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    receptor = meta["receptor"]
    cx, cy, cz = meta["cx"], meta["cy"], meta["cz"]

    if not os.path.exists(receptor):
        print(f"[REFINE][ERROR] receptor not found: {receptor}")
        return

    # --------------------------------------------------
    # 2) Decide which input CSV to use
    #    Prefer TopHits; otherwise fall back to Final_AutoPipeline.csv
    # --------------------------------------------------
    input_csv = None

    top_hits_dir = os.path.join(result_dir, "top_hits")
    if os.path.isdir(top_hits_dir):
        for fn in os.listdir(top_hits_dir):
            if fn.lower().endswith(".csv"):
                input_csv = os.path.join(top_hits_dir, fn)
                break

    if input_csv is None:
        candidate = os.path.join(result_dir, "Final_AutoPipeline.csv")
        if os.path.exists(candidate):
            input_csv = candidate

    if input_csv is None:
        print("[REFINE][ERROR] No input CSV found (TopHits or Final_AutoPipeline.csv)")
        return

    print(f"[REFINE] Using input CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    if "best_pose" not in df.columns:
        print("[REFINE][ERROR] best_pose column not found in CSV")
        return

    # --------------------------------------------------
    # 3) Sort and select refine targets (top 5%)
    # --------------------------------------------------
    if "composite_score" in df.columns:
        df = df.sort_values("composite_score", ascending=False)

    top_n = max(1, int(len(df) * 0.05))
    df_refine = df.head(top_n).copy()

    print(f"[REFINE] Refining top {top_n} compounds")

    # --------------------------------------------------
    # 4) Create the refine directory
    # --------------------------------------------------
    refine_dir = os.path.join(result_dir, "refine")
    os.makedirs(refine_dir, exist_ok=True)

    # --------------------------------------------------
    # 5) Re-dock with GNINA
    # --------------------------------------------------
    refined_rows = []

    for idx, row in df_refine.iterrows():
        pose = row.get("best_pose")

        if not isinstance(pose, str):
            continue

        pose = os.path.abspath(pose)
        if not os.path.exists(pose):
            print(f"[REFINE][SKIP] pose not found: {pose}")
            continue

        out_name = os.path.basename(pose)
        out_path = os.path.join(refine_dir, out_name)

        cmd = [
            GNINA_BIN,
            "-r", receptor,
            "-l", pose,
            "--center_x", str(cx),
            "--center_y", str(cy),
            "--center_z", str(cz),
            "--size_x", "20",
            "--size_y", "20",
            "--size_z", "20",
            "--exhaustiveness", "16",
            "-o", out_path
        ]

        print("[REFINE] Running:", " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
        except Exception as e:
            print(f"[REFINE][ERROR] GNINA failed: {e}")
            continue

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            row["best_pose"] = os.path.abspath(out_path)
            refined_rows.append(row)
        else:
            print(f"[REFINE][FAIL] No output generated for {pose}")

    if not refined_rows:
        print("[REFINE][ERROR] No refined poses generated")
        return

    # --------------------------------------------------
    # 6) Save Refined.csv (all refine results)
    # --------------------------------------------------
    df_out = pd.DataFrame(refined_rows)

    refined_csv = os.path.join(result_dir, "Refined.csv")
    df_out.to_csv(refined_csv, index=False)
    print(f"[REFINE] Saved: {refined_csv}")

    # --------------------------------------------------
    # 7) Save Refined_Top10.csv (top 10%)
    # --------------------------------------------------
    df_top10 = df_out.copy()

    if "composite_score" in df_top10.columns:
        df_top10 = df_top10.sort_values("composite_score", ascending=False)

    top10_n = max(1, int(len(df_top10) * 0.10))
    df_top10 = df_top10.head(top10_n)

    refined_top10_csv = os.path.join(result_dir, "Refined_Top10.csv")
    df_top10.to_csv(refined_top10_csv, index=False)

    print(f"[REFINE] Saved: {refined_top10_csv}")

# ------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python refine_engine.py <result_dir>")
        sys.exit(1)

    main(sys.argv[1])