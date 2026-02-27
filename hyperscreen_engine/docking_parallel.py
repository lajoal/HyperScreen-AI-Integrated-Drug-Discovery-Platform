import os
import subprocess
import pandas as pd
from multiprocessing import Pool
from config import GNINA_PATH, DOCKING_PROCESSES, DOCKING_TIMEOUT

def parse_gnina_all_scores(pdbqt_file):
    scores = {"vina_score": 0.0, "vina_norm": 0.0, "cnn_score": 0.0, "tox_score": 0.0}
    if not os.path.exists(pdbqt_file): return scores
    with open(pdbqt_file, "r") as f:
        for line in f:
            if "REMARK minimizedAffinity" in line:
                try: scores["vina_score"] = float(line.split()[-1])
                except: pass
            elif "REMARK CNNscore" in line:
                try: scores["cnn_score"] = float(line.split()[-1])
                except: pass
            elif "REMARK CNNaffinity" in line:
                try: scores["tox_score"] = float(line.split()[-1])
                except: pass
    return scores

def dock_one(args):
    idx, ligand, name, receptor, cx, cy, cz, result_dir = args
    if os.path.exists(os.path.join(result_dir, "stop.flag")): return idx, None, None
    out_file = os.path.join(result_dir, f"{name}_out.pdbqt")
    if os.path.exists(out_file): return idx, parse_gnina_all_scores(out_file), out_file
    
    cmd = [GNINA_PATH, "-r", receptor, "-l", ligand, "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
           "--size_x", "20", "--size_y", "20", "--size_z", "20", "--exhaustiveness", "8", "-o", out_file]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=DOCKING_TIMEOUT)
    except: return idx, None, None
    return idx, parse_gnina_all_scores(out_file), out_file

def run_docking_parallel(df, receptor, cx, cy, cz, result_dir):
    for col in ["vina_score", "vina_norm", "cnn_score", "tox_score"]: df[col] = 0.0
    df["best_pose"] = None
    
    tasks = []
    for idx, row in df.iterrows():
        ligand = row.get("ligand_path")
        # [Fix] Skip header strings or invalid paths
        if pd.isna(ligand) or "ligand_path" in str(ligand) or not os.path.exists(str(ligand)):
            continue
        tasks.append((idx, ligand, f"ID_{idx}", receptor, cx, cy, cz, result_dir))
        
    with Pool(processes=DOCKING_PROCESSES) as pool:
        for idx, scores, pose in pool.imap_unordered(dock_one, tasks):
            if os.path.exists(os.path.join(result_dir, "stop.flag")): break
            if scores:
                df.at[idx, "vina_score"], df.at[idx, "cnn_score"], df.at[idx, "tox_score"], df.at[idx, "best_pose"] = \
                scores["vina_score"], scores["cnn_score"], scores["tox_score"], pose
    return df