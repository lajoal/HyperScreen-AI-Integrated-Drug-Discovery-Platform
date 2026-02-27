# ======================================================
# IMPORT
# ======================================================
import streamlit as st
import pandas as pd
import os
import numpy as np
import subprocess
import requests
import time
import json
from streamlit_molstar import st_molstar
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors

# ======================================================
# PATH setup
# ======================================================
# By default, paths are resolved from the repository root (where app.py lives).
# You can override them with environment variables.
#   HYPERSCREEN_BASE_DIR: project root (default: app.py directory)
#   HYPERSCREEN_DATA_DIR: data root (default: <BASE_DIR>/data)
import sys
from pathlib import Path

BASE_DIR = Path(os.environ.get("HYPERSCREEN_BASE_DIR", Path(__file__).resolve().parent)).resolve()
DATA_ROOT = Path(os.environ.get("HYPERSCREEN_DATA_DIR", BASE_DIR / "data")).resolve()
RAW_DATA_DIR = DATA_ROOT / "protein_chemical_raw_data"
DATA_DIR = DATA_ROOT / "protein_chemical"

ENGINE_PYTHON = os.environ.get("HYPERSCREEN_PYTHON", sys.executable)
ENGINE_SCRIPT = str(BASE_DIR / "hyperscreen_engine" / "orchestrator.py")
REFINE_SCRIPT = str(BASE_DIR / "hyperscreen_engine" / "refine_engine.py")

# ======================================================
# PDBQT -> PDB (for receptor)
# ======================================================
def pdbqt_to_pdb_text(pdbqt_path: str) -> str:
    lines = []
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                lines.append(line[:66].rstrip() + "\n")
    lines.append("END\n")
    return "".join(lines)

# ======================================================
# [FIX] PDBQT -> single ligand (compatible with GNINA output)
# ======================================================
def pdbqt_best_pose_single_ligand(pdbqt_path: str) -> str:
    if not os.path.exists(pdbqt_path):
        return ""
    has_model = False
    first_model_started = False
    in_first_model = False
    model_lines = []
    fallback_lines = []
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                has_model = True
                if not first_model_started:
                    first_model_started = True
                    in_first_model = True
                else:
                    in_first_model = False
                continue
            if has_model:
                if in_first_model and line.startswith("ENDMDL"):
                    break
                if in_first_model and line.startswith(("ATOM", "HETATM")):
                    model_lines.append(line[:66].rstrip() + "\n")
            else:
                if line.startswith(("ATOM", "HETATM")):
                    fallback_lines.append(line[:66].rstrip() + "\n")
    final_lines = model_lines if model_lines else fallback_lines
    if not final_lines:
        return ""
    final_lines.append("END\n")
    return "".join(final_lines)

# ======================================================
# SESSION STATE (stronger refresh logic)
# ======================================================
# Generate a unique session ID (prevents stale browser rendering)
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(time.time())

defaults = {
    "p_center": "0.0, 0.0, 0.0",
    "search_results": [],
    "current_target": None,
    "receptor_path": None,
    "current_result_dir": None,
    "library_path": None,
    "active_tab": "🎯 Target Setup",
    "last_loaded_target": None,
    "view_nonce": 0,  # [ADDED] Nonce to prevent stale 3D rendering
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# Hard reset: clear data/resource cache and refresh the session ID
def hard_refresh_target(new_target_name):
    if st.session_state.get("last_loaded_target") != new_target_name:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state["session_id"] = str(time.time())  # Force widget ID changes
        st.session_state["last_loaded_target"] = new_target_name
        st.session_state["receptor_path"] = None  # Block stale viewer before searching

# ======================================================
# [NEW] Persist / Restore session state via query params
# ======================================================
def persist_state_to_query():
    qp = st.query_params
    if st.session_state.get("current_result_dir"):
        qp["result_dir"] = st.session_state["current_result_dir"]
    if st.session_state.get("receptor_path"):
        qp["receptor_path"] = st.session_state["receptor_path"]
    if st.session_state.get("library_path"):
        qp["library_path"] = st.session_state["library_path"]
    if st.session_state.get("active_tab"):
        qp["active_tab"] = st.session_state["active_tab"]

def restore_state_from_query():
    qp = st.query_params

    def _get(key: str):
        v = qp.get(key, None)
        if isinstance(v, list):
            return v[0] if v else None
        return v

    if not st.session_state.get("current_result_dir"):
        v = _get("result_dir")
        if v and os.path.exists(v):
            st.session_state["current_result_dir"] = v
    if not st.session_state.get("receptor_path"):
        v = _get("receptor_path")
        if v and os.path.exists(v):
            st.session_state["receptor_path"] = v
    if not st.session_state.get("library_path"):
        v = _get("library_path")
        if v and os.path.exists(v):
            st.session_state["library_path"] = v

    v = _get("active_tab")
    if v in ["🎯 Target Setup", "📊 Results", "🧬 FINAL CANDIDATES (2D STRUCTURE)"]:
        st.session_state["active_tab"] = v

restore_state_from_query()

# ======================================================
# UTIL (preserve original behavior)
# ======================================================
def auto_center(txt: str) -> str:
    coords = []
    for l in txt.splitlines():
        if l.startswith(("ATOM", "HETATM")):
            try:
                coords.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            except:
                pass
    if not coords:
        return "0.0, 0.0, 0.0"
    c = np.mean(coords, axis=0)
    return f"{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}"

def gpu_status() -> str:
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,nounits,noheader"]
            )
            .decode()
            .strip()
            .split(",")
        )
        return f"🟢 GPU {out[0].strip()}% | VRAM {out[1].strip()}/{out[2].strip()} MB"
    except:
        return "GPU info not available"

# ======================================================
# Helper: opt_rmsd display + download normalization
# ======================================================
def normalize_opt_rmsd_for_display(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer opt_rmsd_display (if provided by orchestrator).
    Convert sentinel (<=0.0011) to NaN for UI.
    """
    df = df_in.copy()

    if "opt_rmsd_display" in df.columns:
        df["opt_rmsd"] = df["opt_rmsd_display"]
        df = df.drop(columns=["opt_rmsd_display"])

    if "opt_rmsd" in df.columns:
        df["opt_rmsd"] = pd.to_numeric(df["opt_rmsd"], errors="coerce")
        df.loc[df["opt_rmsd"] <= 0.0011, "opt_rmsd"] = np.nan

    return df

def dataframe_to_download_bytes(df_in: pd.DataFrame) -> bytes:
    """
    Apply the same normalization logic as UI, then return CSV bytes for download.
    (NaN becomes blank cells in CSV)
    """
    df = normalize_opt_rmsd_for_display(df_in)
    return df.to_csv(index=False).encode("utf-8")

# ======================================================
# 3D Viewer Dialog
# ======================================================
@st.dialog("3D Protein–Ligand Interaction", width="large")
def show_3d_interaction(receptor_path: str, pose_path: str, compound_name: str):
    if not (receptor_path and pose_path):
        st.error("Invalid structure path.")
        return
    if not (os.path.exists(receptor_path) and os.path.exists(pose_path)):
        st.error("Structure file missing.")
        return

    receptor_txt = pdbqt_to_pdb_text(receptor_path) if receptor_path.lower().endswith(".pdbqt") else open(receptor_path).read()
    ligand_txt = pdbqt_best_pose_single_ligand(pose_path)
    if not ligand_txt.strip():
        st.error("Ligand pose not found.")
        return

    tmp_dir = os.path.join(os.path.dirname(pose_path), "_tmp_view")
    os.makedirs(tmp_dir, exist_ok=True)
    safe_name = "".join([c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(compound_name)])[:80]

    # [CHANGED] Include nonce in the filename to avoid caching/stale rendering
    nonce = int(st.session_state.get("view_nonce", 0))
    complex_pdb = os.path.join(tmp_dir, f"complex_{safe_name}_{nonce}.pdb")

    with open(complex_pdb, "w") as f:
        f.write(receptor_txt.rstrip() + "\n" + ligand_txt)

    # [CHANGED] Include nonce in the key as well
    st_molstar(complex_pdb, height=600, key=f"dlg_{safe_name}_{st.session_state.session_id}_{nonce}")
    st.caption("Protein + top-ranked docking pose (first model)")

# ======================================================
# UI layout and tab logic
# ======================================================
st.set_page_config(layout="wide", page_title="HyperScreen AI")
st.title("🔬 HyperScreen AI – Integrated Drug Discovery Platform")

tabs = ["🎯 Target Setup", "📊 Results", "🧬 FINAL CANDIDATES (2D STRUCTURE)"]
selected_tab = st.radio(
    "Navigation",
    tabs,
    horizontal=True,
    index=tabs.index(st.session_state.active_tab) if st.session_state.active_tab in tabs else 0,
)
st.session_state.active_tab = selected_tab
persist_state_to_query()

# ======================================================
# TAB 1 - TARGET SETUP
# ======================================================
if selected_tab == "🎯 Target Setup":
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        # [ADDED] Force full session reset button (sidebar)
        if st.sidebar.button("🔄 Clear All Structures"):
            st.query_params.clear()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.receptor_path = None
            st.session_state.current_target = None
            st.session_state.session_id = str(time.time())
            st.rerun()

        mode = st.radio("Mode", ["AlphaFold DB 검색", "PDB 업로드"], horizontal=True)
        if mode == "PDB 업로드":
            uploaded = st.file_uploader("Upload PDB", type=["pdb"])
            if uploaded:
                # Reset cache on new file upload
                hard_refresh_target(uploaded.name)
                txt = uploaded.getvalue().decode()
                path = str(DATA_DIR / uploaded.name)
                os.makedirs(str(DATA_DIR), exist_ok=True)
                with open(path, "w") as f:
                    f.write(txt)
                st.session_state.current_target = uploaded.name
                st.session_state.receptor_path = path
                st.session_state.p_center = auto_center(txt)
                persist_state_to_query()
        else:
            gene = st.text_input("Protein name")
            if st.button("🔍 UniProt Search"):
                # Clear previous viewer path when starting a search (prevents stale view)
                st.session_state.receptor_path = None
                r = requests.get(
                    f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene}+AND+organism_id:9606&format=json&size=5"
                )
                if r.ok:
                    st.session_state.search_results = r.json().get("results", [])
            if st.session_state.search_results:
                acc = st.selectbox("Select protein", [r["primaryAccession"] for r in st.session_state.search_results])
                if st.button("🚀 Load AlphaFold"):
                    # Reset cache when loading AlphaFold structures
                    hard_refresh_target(acc)
                    api = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
                    res = requests.get(api)
                    if res.ok and res.json():
                        pdb_url = res.json()[0]["pdbUrl"]
                        txt = requests.get(pdb_url).text
                        path = str(DATA_DIR / f"{acc}.pdb")
                        os.makedirs(str(DATA_DIR), exist_ok=True)
                        with open(path, "w") as f:
                            f.write(txt)
                        st.session_state.current_target = acc
                        st.session_state.receptor_path = path
                        st.session_state.p_center = auto_center(txt)
                        persist_state_to_query()
                        st.rerun()

    with col_right:
        # [CORE FIX] Render only when receptor_path exists and the file is present
        if st.session_state.receptor_path and os.path.exists(st.session_state.receptor_path):
            st.subheader(f"3D Structure: {st.session_state.current_target}")
            # Combine target name and session ID in the key to force re-rendering
            st_molstar(
                st.session_state.receptor_path,
                height=400,
                key=f"main_v_{st.session_state.current_target}_{st.session_state.session_id}",
            )
        else:
            # Guidance shown before search or after reset
            st.info("단백질을 검색하거나 업로드하면 여기에 3D 구조가 나타납니다.")

        st.session_state.p_center = st.text_input("Pocket Center (x, y, z)", st.session_state.p_center)
        lib = st.radio("Library", ["FDA", "ZINC10K", "ChEMBL50K"], horizontal=True)
        lib_map = {"FDA": "library_3000.csv", "ZINC10K": "zinc_library_10,000_full.csv", "ChEMBL50K": "chembl_subset.csv"}
        lib_path = str(RAW_DATA_DIR / lib_map[lib])

        if st.button("🚀 Start HTVS"):
            if not st.session_state.receptor_path or not os.path.exists(st.session_state.receptor_path):
                st.error("Receptor is not set. Please upload PDB or load AlphaFold first.")
                st.stop()
            result_dir = str(RAW_DATA_DIR / f"{st.session_state.current_target}_{lib}_{int(time.time())}")
            os.makedirs(result_dir, exist_ok=True)
            st.session_state.current_result_dir = result_dir
            st.session_state.library_path = lib_path
            cx, cy, cz = [x.strip() for x in st.session_state.p_center.split(",")]
            with open(os.path.join(result_dir, "meta.json"), "w") as f:
                json.dump({"receptor": st.session_state.receptor_path, "cx": cx, "cy": cy, "cz": cz}, f)
            persist_state_to_query()
            subprocess.Popen(
                [ENGINE_PYTHON, ENGINE_SCRIPT, st.session_state.receptor_path, cx, cy, cz, lib_path, result_dir],
                cwd=str(BASE_DIR / "hyperscreen_engine"),
            )
            st.session_state.active_tab = "📊 Results"
            persist_state_to_query()
            st.rerun()

# ======================================================
# TAB 2 - RESULTS (opt_rmsd-aware UI)
# ======================================================
elif selected_tab == "📊 Results":
    result_dir = st.session_state.get("current_result_dir")
    if not result_dir:
        restore_state_from_query()
        result_dir = st.session_state.get("current_result_dir")

    if not result_dir:
        st.warning("분석 결과 디렉토리를 찾을 수 없습니다.")
        st.stop()

    st.subheader("HTVS Monitor")
    st.info(gpu_status())

    # 1) Load file paths and settings
    meta_path = os.path.join(result_dir, "meta.json")
    lib_path = st.session_state.get("library_path")

    if (not lib_path or not os.path.exists(str(lib_path))) and os.path.exists(meta_path):
        with open(meta_path) as f:
            lib_path = json.load(f).get("library_path")
            st.session_state["library_path"] = lib_path

    # 2) Compute actual_total (effective number of items)
    prep_dir = os.path.join(result_dir, "ligands")
    prep_done = len([f for f in os.listdir(prep_dir) if f.endswith(".pdb")]) if os.path.exists(prep_dir) else 0

    # Denominator consistency: prefer prepped-file count, but set minimum to 1 to avoid /0
    actual_total = prep_done
    if actual_total == 0 and lib_path and os.path.exists(lib_path):
        try:
            actual_total = len(pd.read_csv(lib_path))
        except:
            actual_total = 0
    actual_total = max(actual_total, 1)

    # 3) Aggregate per-step progress
    dock_done = len([f for f in os.listdir(result_dir) if f.endswith("_out.pdbqt")])

    # [CORE FIX] Track progress using opt_rmsd_display (computed only) instead of opt_rmsd sentinel
    opt_total = 10
    opt_done = 0
    final_path = os.path.join(result_dir, "Final_AutoPipeline.csv")

    if os.path.exists(final_path):
        try:
            temp_df = pd.read_csv(final_path)
            if "opt_rmsd_display" in temp_df.columns:
                opt_done = int(temp_df["opt_rmsd_display"].notna().sum())
                opt_done = min(opt_done, opt_total)
            elif "md_score" in temp_df.columns:
                # fallback: md_score > 0 means computed
                opt_done = int((pd.to_numeric(temp_df["md_score"], errors="coerce") > 0).sum())
                opt_done = min(opt_done, opt_total)
        except:
            pass

    # If CSV is not created yet (or not updated), supplement via folder scanning
    if opt_done == 0:
        md_dir = os.path.join(result_dir, "md")
        if os.path.exists(md_dir):
            opt_done = len(os.listdir(md_dir))
            opt_done = min(opt_done, opt_total)

    # 4) UI display - three independent gauges (terminology aligned)
    st.markdown("---")
    c1_1, c1_2 = st.columns([1, 4])
    c1_1.metric("1. Prep", f"{prep_done} / {actual_total}")
    c1_2.progress(min(prep_done / actual_total, 1.0), text="Ligand Preparation")

    c2_1, c2_2 = st.columns([1, 4])
    c2_1.metric("2. Docking", f"{dock_done} / {actual_total}")
    c2_2.progress(min(dock_done / actual_total, 1.0), text="Molecular Docking (GNINA)")

    c3_1, c3_2 = st.columns([1, 4])
    c3_1.metric("3. Opt", f"{opt_done} / {opt_total}")
    c3_2.progress(min(opt_done / opt_total, 1.0), text="Ligand Geometry Optimization (Top 10)")
    st.markdown("---")

    # 5) Control buttons (Stop / Resume / Refine)
    stop_flag_path = os.path.join(result_dir, "stop.flag")
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if os.path.exists(stop_flag_path):
            if st.button("▶️ Resume Pipeline", use_container_width=True, type="primary"):
                os.remove(stop_flag_path)
                with open(meta_path) as f:
                    m = json.load(f)
                subprocess.Popen(
                    [ENGINE_PYTHON, ENGINE_SCRIPT, str(m["receptor"]), str(m["cx"]), str(m["cy"]), str(m["cz"]), lib_path, result_dir],
                    cwd=str(BASE_DIR / "hyperscreen_engine"),
                )
                st.rerun()
        else:
            if st.button("🛑 Stop Pipeline", use_container_width=True):
                with open(stop_flag_path, "w") as f:
                    f.write("stop")
                st.rerun()

    with btn_col2:
        if os.path.exists(final_path):
            # [FIX] Download normalized CSV (opt_rmsd sentinel -> NA; use opt_rmsd_display if present)
            df_dl = pd.read_csv(final_path)
            csv_bytes = dataframe_to_download_bytes(df_dl)
            st.download_button(
                "📥 Download Result",
                data=csv_bytes,
                file_name="Final_AutoPipeline.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("📥 Result not ready", disabled=True, use_container_width=True)

    with btn_col3:
        if st.button("🧬 Refine Top 5%", use_container_width=True):
            subprocess.Popen([ENGINE_PYTHON, REFINE_SCRIPT, result_dir], cwd=str(BASE_DIR / "hyperscreen_engine"))
            st.session_state.active_tab = "🧬 FINAL CANDIDATES (2D STRUCTURE)"
            persist_state_to_query()
            st.rerun()

    # 6) Results dataframe
    if os.path.exists(final_path):
        df_show = pd.read_csv(final_path)
        df_show = normalize_opt_rmsd_for_display(df_show)
        st.dataframe(df_show, height=400)
    else:
        if not os.path.exists(stop_flag_path):
            time.sleep(5)
            st.rerun()

# ======================================================
# TAB 3 - FINAL CANDIDATES (extended features)
# ======================================================
elif selected_tab == "🧬 FINAL CANDIDATES (2D STRUCTURE)":
    result_dir = st.session_state.get("current_result_dir")
    if not result_dir:
        restore_state_from_query()
        result_dir = st.session_state.get("current_result_dir")

    if not result_dir:
        st.warning("No active result directory. Please run HTVS first.")
        st.stop()

    final_path = next(
        (os.path.join(result_dir, f) for f in ["Refined_with_MD.csv", "Refined.csv", "Final_AutoPipeline.csv"] if os.path.exists(os.path.join(result_dir, f))),
        None,
    )
    if not final_path:
        st.warning("No result CSV found yet.")
        st.stop()

    df = pd.read_csv(final_path)

    # 1) Add Top % slider
    st.subheader("📦 Result Export & Filtering")
    top_percent = st.slider("Select Top % to display/export", 1, 100, 10)

    # Filter by ranking
    if "rank" not in df.columns and "composite_score" in df.columns:
        df = df.sort_values("composite_score", ascending=False)
        df["rank"] = np.arange(1, len(df) + 1)

    cutoff = max(1, int(len(df) * (top_percent / 100)))
    df_top = df[df["rank"] <= cutoff] if "rank" in df.columns else df.head(cutoff)

    # 2) Unified toolbox (SDF export, report generation, ZIP download)
    col_tools1, col_tools2, col_tools3 = st.columns(3)

    with col_tools1:
        if st.button("💾 Generate SDF (Top %)", use_container_width=True):
            sdf_path = os.path.join(result_dir, f"Top_{top_percent}pc_candidates.sdf")
            writer = Chem.SDWriter(sdf_path)
            for _, row in df_top.iterrows():
                mol = Chem.MolFromSmiles(str(row["SMILES"]))
                if mol:
                    mol.SetProp("_Name", str(row.get("Compound_Name", "ID")))
                    for col in df.columns:
                        mol.SetProp(col, str(row[col]))
                    writer.write(mol)
            writer.close()
            with open(sdf_path, "rb") as f:
                st.download_button("📥 Download SDF", f, file_name=os.path.basename(sdf_path), use_container_width=True)

    with col_tools2:
        # [FIX] Generate Report.csv with the same opt_rmsd normalization used in UI/download
        if st.button("📊 Generate Report.csv", use_container_width=True):
            report_path = os.path.join(result_dir, "screening_report.csv")
            df_rep = df_top.copy()
            df_rep = normalize_opt_rmsd_for_display(df_rep)
            df_rep.to_csv(report_path, index=False)
            with open(report_path, "rb") as f:
                st.download_button("📥 Download Report", f, file_name="screening_report.csv", use_container_width=True)

    with col_tools3:
        if st.button("📁 Pack All to ZIP", use_container_width=True):
            import zipfile
            zip_path = os.path.join(result_dir, "HTVS_Full_Results.zip")
            with zipfile.ZipFile(zip_path, "w") as zipf:
                zipf.write(final_path, arcname=os.path.basename(final_path))
                for _, row in df_top.iterrows():
                    pose = row.get("best_pose")
                    if pose and os.path.exists(str(pose)):
                        zipf.write(pose, arcname=f"poses/{os.path.basename(str(pose))}")
            with open(zip_path, "rb") as f:
                st.download_button("📥 Download HTVS ZIP", f, file_name="HTVS_Full_Results.zip", use_container_width=True)

    st.divider()

    # --- Compound list rendering loop (preserve original behavior) ---
    for i, row in df_top.iterrows():
        compound_id = row.get("Compound_Name") if isinstance(row.get("Compound_Name"), str) and row.get("Compound_Name").strip() else f"ID_{i}"
        smiles = row.get("SMILES")
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None

        if mol:
            with st.container(border=True):
                c1, c2 = st.columns([1.2, 2])
                with c1:
                    st.image(Draw.MolToImage(mol, size=(300, 300)))
                    pose = row.get("best_pose")
                    if isinstance(pose, str) and os.path.exists(pose):
                        if st.button(f"🔍 3D Interaction: {compound_id}", key=f"v_{i}"):
                            st.session_state["view_nonce"] += 1
                            persist_state_to_query()
                            show_3d_interaction(st.session_state.receptor_path, pose, compound_id)
                    else:
                        st.caption("No docking pose file.")
                with c2:
                    st.subheader(compound_id)
                    st.text(f"Formula: {rdMolDescriptors.CalcMolFormula(mol)}")

                    # Prefer display value if present
                    rmsd_val = row.get("opt_rmsd_display", row.get("opt_rmsd", np.nan))
                    try:
                        rmsd_val = float(rmsd_val)
                    except:
                        rmsd_val = np.nan

                    m1, m2 = st.columns(2)
                    m1.metric("Composite Score", f"{row.get('composite_score', 0):.4f}")

                    if ("opt_rmsd" in df.columns) or ("opt_rmsd_display" in df.columns):
                        if pd.isna(rmsd_val) or rmsd_val <= 0.0011:
                            m2.metric("Opt RMSD", "N/A")
                        else:
                            m2.metric("Opt RMSD", f"{rmsd_val:.4f}")

                    st.text_area("SMILES", str(smiles), height=60, key=f"smi_{i}")
