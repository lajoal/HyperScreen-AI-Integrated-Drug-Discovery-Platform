# ======================================================
# TOXICITY PREDICTOR (Rule-based ADMET proxy)
# ======================================================

from rdkit import Chem
from rdkit.Chem import Descriptors


def predict_toxicity(smiles: str) -> float:
    """
    Heuristic toxicity risk score (0.0 ~ 1.0)
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 1.0

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    rings = Descriptors.RingCount(mol)

    # Baseline (important)
    risk = 0.05

    if mw > 600:
        risk += 0.25
    elif mw > 500:
        risk += 0.15

    if logp > 5:
        risk += 0.25
    elif logp > 4:
        risk += 0.10

    if hbd > 5:
        risk += 0.15
    if hba > 10:
        risk += 0.15

    if tpsa < 20:
        risk += 0.15
    elif tpsa > 140:
        risk += 0.20

    if rot > 10:
        risk += 0.10
    if rings >= 6:
        risk += 0.10

    return min(round(risk, 3), 1.0)
