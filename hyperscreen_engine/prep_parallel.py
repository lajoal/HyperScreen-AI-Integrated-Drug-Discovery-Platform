# ======================================================
# PREP PARALLEL – WORKING VERSION (DO NOT MODIFY)
# ======================================================

import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import AllChem
from config import PREP_WORKERS

TIMEOUT = 10


def embed_worker(smiles, output_file):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    mol = Chem.AddHs(mol)

    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        return False

    Chem.MolToPDBFile(mol, output_file)
    return True


def safe_prep(idx, smiles, name, ligand_dir):
    output_file = os.path.join(ligand_dir, f"{name}.pdb")

    ctx = mp.get_context("spawn")
    p = ctx.Process(target=embed_worker, args=(smiles, output_file))
    p.start()
    p.join(TIMEOUT)

    if p.is_alive():
        p.terminate()
        p.join()
        return idx, None

    if os.path.exists(output_file):
        return idx, output_file

    return idx, None


def run_prep_parallel(df, ligand_dir):

    os.makedirs(ligand_dir, exist_ok=True)
    df["ligand_path"] = None

    with ThreadPoolExecutor(max_workers=PREP_WORKERS) as executor:

        futures = []

        for idx, row in df.iterrows():
            smiles = row["SMILES"]
            name = f"ID_{idx}"
            futures.append(
                executor.submit(safe_prep, idx, smiles, name, ligand_dir)
            )

        for future in as_completed(futures):
            idx, result = future.result()
            df.at[idx, "ligand_path"] = result

    return df
