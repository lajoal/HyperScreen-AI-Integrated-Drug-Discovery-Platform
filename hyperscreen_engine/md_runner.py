import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import linear_sum_assignment


def kabsch_rmsd(P, Q):
    """
    Kabsch 알고리즘: 두 점 집합 P, Q를 최적으로 겹친 후 RMSD 계산
    (회전/이동 제거)
    """
    if len(P) < 3 or len(Q) < 3:
        return 999.0

    P_center = P - np.mean(P, axis=0)
    Q_center = Q - np.mean(Q, axis=0)

    H = np.dot(P_center.T, Q_center)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    d = np.linalg.det(np.dot(V, U.T))
    step = np.eye(3)
    if d < 0:
        step[2, 2] = -1

    rot = np.dot(V, np.dot(step, U.T))
    P_rot = np.dot(P_center, rot)

    return np.sqrt(np.mean(np.sum((P_rot - Q_center) ** 2, axis=1)))


def _pdbqt_atomtype_to_element(atom_type: str) -> str:
    """
    PDBQT atom type -> element symbol (대략 변환)
    예: C, A, N, OA, SA, HD, Cl, Br ...
    """
    if not atom_type:
        return "C"

    at = atom_type.strip()

    # Prefer common 2-letter elements first
    if at.startswith("Cl"):
        return "Cl"
    if at.startswith("Br"):
        return "Br"
    if at.startswith("Si"):
        return "Si"

    # Normalize AutoDock-style atom types
    # A = aromatic carbon
    if at == "A":
        return "C"
    # OA = oxygen acceptor, NA = nitrogen acceptor, SA = sulfur acceptor
    if at.startswith("OA"):
        return "O"
    if at.startswith("NA"):
        return "N"
    if at.startswith("SA"):
        return "S"
    # HD = polar hydrogen
    if at.startswith("HD"):
        return "H"

    # Default: use the first letter
    first = at[0].upper()
    if first in ["C", "N", "O", "S", "P", "F", "I", "H", "B", "K", "Z", "M"]:
        return first

    return "C"


def _extract_last_model_from_pdbqt(lines):
    """
    PDBQT에서 마지막 MODEL(또는 모델 구분 없으면 전체)만 추출
    반환: 해당 모델의 ATOM/HETATM line 리스트
    """
    all_models = []
    current_lines = []
    saw_model = False

    for l in lines:
        if l.startswith("MODEL"):
            saw_model = True
            current_lines = []
        elif l.startswith(("ATOM", "HETATM")):
            current_lines.append(l)
        elif l.startswith("ENDMDL"):
            if current_lines:
                all_models.append(current_lines)

    # Case without MODEL/ENDMDL blocks
    if not saw_model:
        current_lines = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
        if current_lines:
            all_models.append(current_lines)

    if not all_models:
        return None

    return all_models[-1]


def _parse_pdbqt_heavy_coords_elems(atom_lines):
    """
    PDBQT atom lines -> heavy atom 좌표/원자타입 추출
    반환:
      coords: (N,3) np.array
      elems:  [str, ...]
    """
    coords = []
    elems = []

    for l in atom_lines:
        try:
            x = float(l[30:38])
            y = float(l[38:46])
            z = float(l[46:54])

            # PDBQT atom type is typically the last token
            parts = l.split()
            atom_type = parts[-1] if parts else "C"
            elem = _pdbqt_atomtype_to_element(atom_type)

            # heavy atom only
            if elem == "H":
                continue

            coords.append([x, y, z])
            elems.append(elem)
        except Exception:
            continue

    if len(coords) == 0:
        return None, None

    return np.array(coords, dtype=float), elems


def _get_rdkit_heavy_coords_elems(mol):
    """
    RDKit mol conformer에서 heavy atom 좌표/원소 추출
    반환:
      coords: (N,3) np.array
      elems:  [str, ...]
    """
    conf = mol.GetConformer()
    coords = []
    elems = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:  # Exclude hydrogens
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        elems.append(atom.GetSymbol())

    if len(coords) == 0:
        return None, None

    return np.array(coords, dtype=float), elems


def match_atoms_by_type_and_distance(P, Q, elemP, elemQ, penalty=1e6):
    """
    같은 원자 타입끼리 우선 매칭 + 거리 최소화 (Hungarian)
    P: docking heavy atom coords
    Q: RDKit heavy atom coords
    elemP, elemQ: 각 좌표의 element symbol 리스트
    반환: P와 대응되도록 재정렬된 Q
    """
    if len(P) == 0 or len(Q) == 0:
        return None

    # cost matrix
    dist = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    cost = dist.copy()

    # Apply a large penalty when element types do not match
    for i in range(len(P)):
        for j in range(len(Q)):
            if elemP[i] != elemQ[j]:
                cost[i, j] += penalty

    row_ind, col_ind = linear_sum_assignment(cost)

    # If many matches carry penalties (type mismatch), confidence is lower
    # Still compute RMSD, but return Q in a deterministic order
    Q_matched = Q[col_ind]

    # Ensure ordering by row_ind (usually already 0..N-1)
    if not np.array_equal(row_ind, np.arange(len(row_ind))):
        order = np.argsort(row_ind)
        Q_matched = Q_matched[order]

    return Q_matched


def run_md(pdbqt_path, workdir, receptor=None, smiles=None):
    """
    Pose stability surrogate:
    Docking pose (PDBQT) vs RDKit-optimized conformer (SMILES) RMSD
    - heavy atom only
    - element-aware Hungarian matching
    - Kabsch alignment
    """
    try:
        if not pdbqt_path or not os.path.exists(pdbqt_path):
            return 0.0001
        if not smiles or not isinstance(smiles, str):
            return 0.0001

        # --------------------------------------------------
        # 1) Extract heavy-atom coordinates/elements from the last model in PDBQT
        # --------------------------------------------------
        with open(pdbqt_path, "r") as f:
            lines = f.readlines()

        last_model_lines = _extract_last_model_from_pdbqt(lines)
        if not last_model_lines:
            return 0.0001

        ref_coords, ref_elems = _parse_pdbqt_heavy_coords_elems(last_model_lines)
        if ref_coords is None or len(ref_coords) < 3:
            return 0.0001

        # --------------------------------------------------
        # 2) Build RDKit 3D from SMILES and optimize
        # --------------------------------------------------
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0001

        mol = Chem.AddHs(mol)

        # ETKDG embedding (fallback on failure)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_status = AllChem.EmbedMolecule(mol, params)

        if embed_status != 0:
            # fallback
            embed_status = AllChem.EmbedMolecule(mol, randomSeed=42)
            if embed_status != 0:
                return 0.0001

        # Prefer MMFF; fall back to UFF
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            # If force-field optimization fails, proceed with embedded coordinates
            pass

        opt_coords, opt_elems = _get_rdkit_heavy_coords_elems(mol)
        if opt_coords is None or len(opt_coords) < 3:
            return 0.0001

        # --------------------------------------------------
        # 3) Handle atom-count differences
        #    (should match in principle, but guard against parsing/protonation differences)
        # --------------------------------------------------
        # Optionally filter to shared atoms by element counts
        # Note: Hungarian assignment supports rectangular cost matrices
        # (scipy.optimize.linear_sum_assignment supports rectangular matrices)

        # --------------------------------------------------
        # 4) element-aware matching + Kabsch RMSD
        # --------------------------------------------------
        Q_matched = match_atoms_by_type_and_distance(
            ref_coords, opt_coords, ref_elems, opt_elems, penalty=1e6
        )
        if Q_matched is None or len(Q_matched) < 3:
            return 0.0001

        # Length of P and Q_matched follows Hungarian result: min(len(P), len(Q))
        # For rectangular matrices, len(row_ind) = min(n_rows, n_cols)
        n = min(len(ref_coords), len(Q_matched))
        if n < 3:
            return 0.0001

        P = ref_coords[:n]
        Q = Q_matched[:n]

        rmsd = kabsch_rmsd(P, Q)

        # Apply floor to preserve the existing interface
        return round(float(rmsd), 4) if rmsd > 0.0005 else 0.001

    except Exception:
        return 0.0001