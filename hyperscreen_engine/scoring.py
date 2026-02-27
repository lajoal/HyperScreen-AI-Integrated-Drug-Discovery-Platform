import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

# ------------------------------------------------------
# TOXICITY PREDICTOR (heuristic ADMET, screening-use)
# ------------------------------------------------------
def safe_toxicity(smiles):
    """
    Heuristic toxicity risk score (0.0 ~ 1.0)

    NOTE
    ----
    This is a screening-stage ADMET proxy used for compound prioritization.
    It does NOT represent clinical toxicity.
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

    # baseline (avoid all-zero collapse)
    risk = 0.05

    # molecular size
    if mw > 600:
        risk += 0.25
    elif mw > 500:
        risk += 0.15

    # lipophilicity
    if logp > 5:
        risk += 0.25
    elif logp > 4:
        risk += 0.10

    # hydrogen bonding burden
    if hbd > 5:
        risk += 0.15
    if hba > 10:
        risk += 0.15

    # polarity / permeability
    if tpsa < 20:
        risk += 0.15
    elif tpsa > 140:
        risk += 0.20

    # molecular complexity
    if rot > 10:
        risk += 0.10
    if rings >= 6:
        risk += 0.10

    return min(round(risk, 3), 1.0)


# ------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------
def normalize(x):
    """
    Min-max normalization.

    NOTE
    ----
    If all values are identical, return zeros to preserve
    relative ranking stability in early-stage screening.
    """
    x = np.array(x, dtype=float)
    if len(x) == 0:
        return x
    min_v, max_v = np.nanmin(x), np.nanmax(x)
    if max_v - min_v == 0:
        return np.zeros_like(x)
    return (x - min_v) / (max_v - min_v)


# ------------------------------------------------------
# COMPOSITE SCORE
# ------------------------------------------------------
def compute_composite(df):
    """
    Composite scoring function (legacy-compatible).

    IMPORTANT INTERPRETATION
    ------------------------
    Although legacy variable names are retained for pipeline compatibility,
    the actual meanings are as follows:

    - vina_score :
        GNINA minimized affinity (physical docking score),
        NOT AutoDock Vina output.

    - vina_norm :
        Normalized GNINA minimized affinity
        (lower affinity → higher normalized score).

    - cnn_score :
        GNINA CNN-based pose score (pose likelihood).

    - cnn_norm :
        Normalized CNN-based pose score.

    - tox_score :
        Heuristic AI-based ADMET toxicity risk score.

    - tox_norm :
        Normalized toxicity score
        (lower toxicity → higher normalized score).

    The composite score integrates physical docking,
    pose quality, and ADMET risk for compound prioritization.
    """

    df = df.copy()

    # ---- DOCKING (GNINA physical affinity) ----
    df["vina_score"] = pd.to_numeric(df["vina_score"], errors="coerce")
    df = df.dropna(subset=["vina_score"])

    # lower (more negative) GNINA affinity is better
    df["vina_norm"] = normalize(-df["vina_score"])

    # ---- CNN (pose quality) ----
    if "cnn_score" not in df.columns:
        df["cnn_score"] = 0.0
    df["cnn_norm"] = normalize(df["cnn_score"])

    # ---- TOXICITY (ADMET proxy) ----
    if "tox_score" not in df.columns:
        df["tox_score"] = df["SMILES"].apply(safe_toxicity)

    # lower toxicity risk is better
    df["tox_norm"] = normalize(-df["tox_score"])

    # ---- COMPOSITE INTEGRATION ----
    # Weighted integration of:
    #   - GNINA physical docking score
    #   - CNN-based pose quality
    #   - AI-based toxicity risk
    df["composite_score"] = (
        0.6 * df["vina_norm"]
        + 0.2 * df["cnn_norm"]
        + 0.2 * df["tox_norm"]
    )

    return df