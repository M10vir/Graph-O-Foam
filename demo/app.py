import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from datetime import datetime

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH for Streamlit
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.synth.generate_frames import generate_sequence
from src.tasks.extract_dynamics import run as extract_dynamics

st.set_page_config(page_title="Foam Stability Copilot", layout="wide")
st.title("🫧 Foam Stability Copilot")
st.caption("Select BD/HD XLSX → generate synthetic microscopy frames → extract bubble dynamics → stability forecast")

DATA_SHEETS = Path("data/sheets")
DATA_SYNTH = Path("data/synth")
DATA_SHEETS.mkdir(parents=True, exist_ok=True)
DATA_SYNTH.mkdir(parents=True, exist_ok=True)

def list_xlsx():
    return sorted(DATA_SHEETS.glob("*.xlsx"))

def read_img(p: Path):
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

# ---------- Sidebar: Select sheets ----------
st.sidebar.header("1) Select datasheets (.xlsx)")
xlsx = list_xlsx()
if not xlsx:
    st.sidebar.error("No XLSX found in data/sheets/. Copy your sheets there first.")
    st.stop()

bd_candidates = [p for p in xlsx if "BD" in p.name.upper()] or xlsx
hd_candidates = [p for p in xlsx if "HD" in p.name.upper()] or xlsx

bd_path = Path(st.sidebar.selectbox("Pick BD (Structure)", [str(p) for p in bd_candidates]))
hd_path = Path(st.sidebar.selectbox("Pick HD (Height/Volume)", [str(p) for p in hd_candidates]))

st.sidebar.header("2) Generate run")
run_name = st.sidebar.text_input("Run name", value=f"{bd_path.stem.replace(' ','_')}_{datetime.now().strftime('%H%M%S')}")
nframes = st.sidebar.number_input("Frames (0 = ALL)", min_value=0, max_value=5000, value=0, step=10)

out_dir = DATA_SYNTH / run_name

if st.sidebar.button("🚀 Generate frames + extract dynamics", type="primary"):
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

    st.success(f"Done ✅  Half-life: {half_life:.2f}s  Run: {out_dir}")
    st.session_state["run_dir"] = str(out_dir)

# ---------- Sidebar: Select existing run ----------
st.sidebar.header("3) Explore run")
runs = sorted([p for p in DATA_SYNTH.iterdir() if p.is_dir()], key=lambda p: p.name) if DATA_SYNTH.exists() else []
if not runs:
    st.sidebar.info("No runs yet. Generate one above.")
    st.stop()

default_run = st.session_state.get("run_dir", str(runs[-1]))
run_dir = Path(st.sidebar.selectbox("Select run folder", [str(r) for r in runs], index=[str(r) for r in runs].index(default_run) if default_run in [str(r) for r in runs] else len(runs)-1))

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

# ---------- KPIs ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Run", run_dir.name)
k2.metric("Time (s)", f"{t_selected:.0f}")
k3.metric("Bubble count N(t)", f"{int(row['n'])}")
k4.metric("Mean radius (px)", f"{row['r_mean']:.2f}")

if half_life is not None:
    st.info(f"⏱️ Foam half-life (from HD sheet): **{half_life:.2f} s**")

# ---------- Main ----------
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

# ---------- Forecast ----------
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

# ---------- Export ----------
st.subheader("Export")
colA, colB = st.columns(2)
with colA:
    st.download_button("Download bubble_dynamics.csv", data=dyn_csv.read_bytes(), file_name="bubble_dynamics.csv", mime="text/csv")
with colB:
    if meta_csv.exists():
        st.download_button("Download frames_metadata.csv", data=meta_csv.read_bytes(), file_name="frames_metadata.csv", mime="text/csv")
    else:
        st.write("frames_metadata.csv not found")
