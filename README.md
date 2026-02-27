# HyperScreen AI

Integrated high-throughput virtual screening (HTVS) + lightweight ADMET proxy scoring + optional ligand geometry optimization (MD surrogate) with a Streamlit UI.

> **Portability note**
> This repo is prepared for GitHub + Zenodo archiving. Paths are **not hard-coded**.
> Executables and key parameters are configurable via environment variables.

## Repository layout

- `app.py` : Streamlit UI
- `hyperscreen_engine/` : pipeline modules
  - `orchestrator.py` : end-to-end HTVS runner
  - `prep_parallel.py` : ligand preparation (OpenBabel)
  - `docking_parallel.py` : GNINA docking (multiprocessing)
  - `scoring.py` : composite scoring (GNINA + CNN + ADMET proxy)
  - `md_runner.py` : ligand geometry optimization surrogate for top hits

## Quick start

### 1) Create environment (recommended: conda)

```bash
conda create -n hyperscreen python=3.10 -y
conda activate hyperscreen

# RDKit is best installed via conda
conda install -c conda-forge rdkit -y

pip install -r requirements.txt
```

### 2) Install external tools

- **GNINA**: install and ensure `gnina` is in your `PATH`, or set `GNINA_PATH`.
- **OpenBabel**: install and ensure `obabel` is in your `PATH`, or set `OBABEL_PATH`.

### 3) Run Streamlit

```bash
streamlit run app.py
```

## Configuration (paths and knobs)

### App paths

- `HYPERSCREEN_BASE_DIR` (default: directory containing `app.py`)
- `HYPERSCREEN_DATA_DIR` (default: `<BASE_DIR>/data`)

### Engine executables

- `GNINA_PATH` (default: `gnina`)
- `OBABEL_PATH` (default: `obabel`)

### Engine parameters

- `PREP_WORKERS` (default: 12)
- `DOCKING_PROCESSES` (default: 4)
- `DOCKING_TIMEOUT` (default: 20 seconds)
- `FAST_EXHAUST` (default: `8`)
- `TOP_PERCENT_MD` (default: 0.05)
- `HYPERSCREEN_DB` (default: `hyperscreen.db`)

Example:

```bash
export GNINA_PATH=/opt/gnina/gnina
export PREP_WORKERS=24
export DOCKING_PROCESSES=8
streamlit run app.py
```

## Reproducible Zenodo release (recommended)

1. Push this repo to GitHub.
2. In Zenodo: **GitHub** integration → enable the repository.
3. Create a GitHub Release (tag): e.g. `v0.1.0`.
4. Zenodo will automatically archive that release and mint a DOI.
5. Cite the Zenodo DOI in your manuscript (`Code availability`).

## License

See `LICENSE`.
