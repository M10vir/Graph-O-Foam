import sys
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import cv2

# Ensure project root is on path for Streamlit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synth.generate_frames import generate_sequence
from src.tasks.extract_dynamics import run as extract_dynamics

st.set_page_config(page_title="Foam Stability Copilot", layout="wide")
st.title("🫧 Foam Stability Copilot")
st.caption("BD/HD XLSX → synthetic microscopy frames → bubble dynamics → stability forecast")

DATA_SHEETS = Path("data/sheets")
DATA_SYNTH  = Path("data/synth")
DATA_UPLOAD = Path("data/uploads")
for p in [DATA_SHEETS, DATA_SYNTH, DATA_UPLOAD]:
    p.mkdir(parents=True, exist_ok=True)

def list_xlsx(folder: Path):
    return sorted(folder.glob("*.xlsx"))

def list_runs():
    return sorted([p for p in DATA_SYNTH.iterdir() if p.is_dir()], key=lambda p: p.name) if DATA_SYNTH.exists() else []

def read_img(p: Path):
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

def save_uploaded_xlsx(upload, target_dir: Path):
    """Save Streamlit UploadedFile to disk and return path."""
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / upload.name
    out.write_bytes(upload.getbuffer())
    return out

# ---------------- Sidebar: Input Source ----------------
st.sidebar.header("0) Input source")
input_source = st.sidebar.radio(
    "Choose input method",
    ["Option A: Pick from data/sheets", "Option B: Upload BD/HD XLSX"],
    horizontal=False
)

# ---------------- Sidebar: Select BD/HD ----------------
st.sidebar.header("1) Select datasheets (.xlsx)")

bd_path = None
hd_path = None

if input_source.startswith("Option A"):
    xlsx = list_xlsx(DATA_SHEETS)
    if not xlsx:
        st.sidebar.error("No XLSX found in data/sheets/. Copy your sheets there first.")
        st.stop()

    bd_candidates = [p for p in xlsx if "BD" in p.name.upper()] or xlsx
    hd_candidates = [p for p in xlsx if "HD" in p.name.upper()] or xlsx

    bd_path = Path(st.sidebar.selectbox("Pick BD (Structure)", [str(p) for p in bd_candidates]))
    hd_path = Path(st.sidebar.selectbox("Pick HD (Height/Volume)", [str(p) for p in hd_candidates]))

else:
    st.sidebar.caption("Upload exactly 2 files: BD.xlsx (Structure) + HD.xlsx (Height/Volume)")
    bd_up = st.sidebar.file_uploader("Upload BD (Structure) .xlsx", type=["xlsx"], key="bd_up")
    hd_up = st.sidebar.file_uploader("Upload HD (Height/Volume) .xlsx", type=["xlsx"], key="hd_up")

    if bd_up is not None:
        bd_path = save_uploaded_xlsx(bd_up, DATA_UPLOAD)
    if hd_up is not None:
        hd_path = save_uploaded_xlsx(hd_up, DATA_UPLOAD)

    st.sidebar.caption(f"Uploads saved under: {DATA_UPLOAD}/")

# ---------------- Sidebar: Generate Run ----------------
st.sidebar.header("2) Generate run")
default_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if bd_path is not None:
    # nicer default run name based on file stem
    default_name = f"{bd_path.stem.replace(' ','_')}_{datetime.now().strftime('%H%M%S')}"

run_name = st.sidebar.text_input("Run name", value=default_name)
nframes = st.sidebar.number_input("Frames (0 = ALL)", min_value=0, max_value=5000, value=0, step=10)

out_dir = DATA_SYNTH / run_name

if st.sidebar.button("🚀 Generate frames + extract dynamics", type="primary"):
    if bd_path is None or hd_path is None:
        st.sidebar.error("Please provide BOTH BD.xlsx and HD.xlsx.")
    else:
        with st.spinner("Generating synthetic microscopy frames..."):
            half_life = generate_sequence(
                bd_xlsx=str(bd_path),
                hd_xlsx=str(hd_path),
                out_dir=str(out_dir),
                n_frames=int(nframes),
                seed=7
            )
        with st.spinner("Extracting bubble dynamics + overlays..."):
            extract_dynamics(folder=str(out_dir), out_overlays=str(out_dir / "overlays"))

        hl_txt = f"{half_life:.2f}s" if half_life is not None else "N/A"
        st.success(f"Done ✅ Half-life: {hl_txt} • Run: {out_dir}")
        st.session_state["run_dir"] = str(out_dir)
        st.rerun()

# ---------------- Sidebar: Explore Run ----------------
st.sidebar.header("3) Explore run")
runs = list_runs()
if not runs:
    st.sidebar.info("No runs found in data/synth yet. Generate one above.")
    st.stop()

default_run = st.session_state.get("run_dir", str(runs[-1]))
run_strs = [str(r) for r in runs]
idx = run_strs.index(default_run) if default_run in run_strs else len(runs) - 1
run_dir = Path(st.sidebar.selectbox("Select run folder", run_strs, index=idx))

# ---------------- Load Data ----------------
frames_dir = run_dir
overlays_dir = run_dir / "overlays"
dyn_csv = run_dir / "bubble_dynamics.csv"
meta_csv = run_dir / "frames_metadata.csv"

if not dyn_csv.exists():
    st.warning(f"Run missing dynamics: {dyn_csv}. Generate/extract first.")
    st.stop()

df = pd.read_csv(dyn_csv).dropna(subset=["t_s"]).sort_values("t_s")

half_life = None
if meta_csv.exists():
    try:
        m = pd.read_csv(meta_csv)
        if "half_life_s" in m.columns and len(m) > 0:
            half_life = float(m["half_life_s"].iloc[0])
    except Exception:
        pass

# ---------------- Controls ----------------
tmin, tmax = float(df["t_s"].min()), float(df["t_s"].max())
t = st.sidebar.slider("Time (s)", min_value=tmin, max_value=tmax, value=tmin, step=5.0)
row = df.iloc[(df["t_s"] - t).abs().argsort()[:1]].iloc[0]
frame_name = row["frame"]
t_selected = float(row["t_s"])
view_mode = st.sidebar.radio("View", ["Original", "Overlay"], horizontal=True)

img_path = frames_dir / frame_name
overlay_path = overlays_dir / frame_name
img = read_img(img_path) if view_mode == "Original" else read_img(overlay_path)

x = None
if img is not None:
    x = img.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)

# ---------------- KPIs ----------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Run", run_dir.name)
k2.metric("Time (s)", f"{t_selected:.0f}")
k3.metric("Bubble count N(t)", f"{int(row['n'])}")
k4.metric("Mean radius (px)", f"{row['r_mean']:.2f}")

if half_life is not None:
    st.info(f"⏱️ Foam half-life (from HD sheet): **{half_life:.2f} s**")

# ---------------- Main Panels ----------------
c1, c2 = st.columns([1.15, 1])

with c1:
    st.subheader("Microscopy Frame")
    st.caption(f"{frame_name} • {view_mode}")
    if x is not None:
        st.image(x, clamp=True, use_container_width=True)
    else:
        st.warning("Could not load selected frame image.")

with c2:
    st.subheader("Bubble Dynamics")
    chart_df = df[["t_s", "n", "r_mean", "r_std", "circ_mean"]].copy()
    st.line_chart(chart_df.set_index("t_s")[["n", "r_mean"]])
    st.caption("Variability & shape stability")
    st.line_chart(chart_df.set_index("t_s")[["r_std", "circ_mean"]])

# ---------------- Forecast ----------------
st.subheader("Stability Forecast (Lite)")
ts = df["t_s"].values.astype(float)
rs = df["r_mean"].values.astype(float)
valid = np.isfinite(ts) & np.isfinite(rs)

coarsening_rate = 0.0
if valid.sum() > 2:
    coef = np.polyfit(ts[valid], rs[valid], 1)
    coarsening_rate = float(coef[0])

score = float(100 * np.exp(-abs(coarsening_rate) * 15))
score = max(0.0, min(100.0, score))

s1, s2, s3 = st.columns(3)
s1.metric("Coarsening rate (Δr/Δt)", f"{coarsening_rate:.5f} px/s")
s2.metric("Stability Score", f"{score:.1f} / 100")
s3.metric("Half-life label", f"{half_life:.2f} s" if half_life is not None else "N/A")

with st.expander("Explainability (Lite)"):
    st.write(
        "- Faster growth of mean bubble radius → faster coarsening → lower stability.\n"
        "- Rising radius std → widening distribution → instability.\n"
        "- Lower circularity → deformation/merging → instability.\n"
        "Next: train a lightweight regressor across runs (GO vs NGO) to predict half-life."
    )

# ---------------- Export ----------------
st.subheader("Export")
colA, colB = st.columns(2)
with colA:
    st.download_button("Download bubble_dynamics.csv", data=dyn_csv.read_bytes(), file_name="bubble_dynamics.csv", mime="text/csv")
with colB:
    if meta_csv.exists():
        st.download_button("Download frames_metadata.csv", data=meta_csv.read_bytes(), file_name="frames_metadata.csv", mime="text/csv")
    else:
        st.write("frames_metadata.csv not found")

# =========================
# Compare two runs (GO vs NGO / any 2 conditions)
# =========================
st.divider()
st.header("🔁 Compare Two Runs (GO vs NGO / Condition A vs B)")

runs_all = list_runs()
run_paths = [Path(r) for r in runs_all]
run_names = [r.name for r in run_paths]

def load_dynamics(run_path: Path) -> pd.DataFrame:
    p = run_path / "bubble_dynamics.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p).dropna(subset=["t_s"]).sort_values("t_s")
    return df

def compute_coarsening_rate(df: pd.DataFrame) -> float:
    if df.empty or "t_s" not in df.columns or "r_mean" not in df.columns:
        return float("nan")
    x = df["t_s"].astype(float).values
    y = df["r_mean"].astype(float).values
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    slope = float(np.polyfit(x[m], y[m], 1)[0])  # px/s
    return slope

def stability_score_from_rate(rate: float) -> float:
    if not np.isfinite(rate):
        return float("nan")
    score = float(100 * np.exp(-abs(rate) * 15))
    return max(0.0, min(100.0, score))

if len(run_names) < 2:
    st.info("Generate at least 2 runs to enable comparison.")
else:
    cA, cB = st.columns(2)

    with cA:
        runA_name = st.selectbox("Run A", run_names, index=max(0, len(run_names) - 2))
    with cB:
        runB_name = st.selectbox("Run B", run_names, index=len(run_names) - 1)

    runA = DATA_SYNTH / runA_name
    runB = DATA_SYNTH / runB_name

    dfA = load_dynamics(runA)
    dfB = load_dynamics(runB)

    if dfA.empty or dfB.empty:
        st.warning("One of the selected runs is missing bubble_dynamics.csv. Please extract dynamics for both runs.")
    else:
        rateA = compute_coarsening_rate(dfA)
        rateB = compute_coarsening_rate(dfB)
        scoreA = stability_score_from_rate(rateA)
        scoreB = stability_score_from_rate(rateB)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Run A coarsening (Δr/Δt)", "N/A" if not np.isfinite(rateA) else f"{rateA:.5f} px/s")
        m2.metric("Run A stability score", "N/A" if not np.isfinite(scoreA) else f"{scoreA:.1f}/100")
        m3.metric("Run B coarsening (Δr/Δt)", "N/A" if not np.isfinite(rateB) else f"{rateB:.5f} px/s")
        m4.metric("Run B stability score", "N/A" if not np.isfinite(scoreB) else f"{scoreB:.1f}/100")

        # Winner label
        winner = None
        if np.isfinite(scoreA) and np.isfinite(scoreB):
            winner = "Run A" if scoreA > scoreB else ("Run B" if scoreB > scoreA else "Tie")
        if winner:
            st.success(f"🏁 More stable (Lite): **{winner}**")

        # Align on time for clean plotting
        plotA = dfA[["t_s", "n", "r_mean", "circ_mean"]].copy()
        plotB = dfB[["t_s", "n", "r_mean", "circ_mean"]].copy()
        plotA = plotA.rename(columns={"n": "A_n", "r_mean": "A_r_mean", "circ_mean": "A_circ"})
        plotB = plotB.rename(columns={"n": "B_n", "r_mean": "B_r_mean", "circ_mean": "B_circ"})

        merged = pd.merge(plotA, plotB, on="t_s", how="outer").sort_values("t_s")

        p1, p2, p3 = st.columns(3)
        with p1:
            st.subheader("Bubble Count N(t)")
            st.line_chart(merged.set_index("t_s")[["A_n", "B_n"]])
        with p2:
            st.subheader("Mean Radius r_mean(t)")
            st.line_chart(merged.set_index("t_s")[["A_r_mean", "B_r_mean"]])
        with p3:
            st.subheader("Circularity (shape stability)")
            st.line_chart(merged.set_index("t_s")[["A_circ", "B_circ"]])

        with st.expander("Explainability"):
            st.write(
                "- Lower coarsening rate (slower increase in mean radius) generally indicates higher stability.\n"
                "- Falling circularity may indicate deformation/merging.\n"
                "- Use this panel to compare GO vs NGO (or any two formulations) directly."
            )

