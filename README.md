<h1 align="center">Graph‑O‑Foam: ActiveScan Copilot (Lite)</h1>
<h3 align="center">XLSX → Frames → Bubble Dynamics → Stability → Comparison</h3>

<p align="center">
  Microscopy Hackathon 2025 — AISCIA Use Case
</p>

<p align="center">

  <!-- Python -->
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />

  <!-- Streamlit -->
  <img src="https://img.shields.io/badge/Streamlit-Framework-FF4B4B?logo=streamlit&logoColor=white" />

  <!-- Pandas -->
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white" />

  <!-- OpenPyXL -->
  <img src="https://img.shields.io/badge/OpenPyXL-Excel%20Integration-1F6E43?logo=microsoft-excel&logoColor=white" />

  <!-- NumPy -->
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white" />

  <!-- OpenCV -->
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white" />

  <!-- Git -->
  <img src="https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git&logoColor=white" />

  <!-- GitHub -->
  <img src="https://img.shields.io/badge/GitHub-Repo-181717?logo=github&logoColor=white" />

</p>

# Graph-O-Foam: ActiveScan Copilot (Lite)
Microscopy Hackathon 2025 (AISCIA Use Case)  
From XLSX datasheets → microscopy-like frames → bubble dynamics → stability forecast + GO vs NGO comparison

---

## Problem
Foam stability is driven by bubble evolution (coarsening, merging, deformation). In practice, teams often don’t have clean, labeled microscopy datasets ready for ML—sometimes they only have BD/HD datasheets (XLSX). We need a fast, explainable workflow to quantify bubble patterns over time and compare stability across formulations (e.g., GO vs NGO).

---

## What this project does
Graph-O-Foam builds a reproducible pipeline that:

1) Ingests BD + HD datasheets (XLSX)
- BD: bubble statistics vs time (drives frame synthesis)
- HD: foam stability signal (e.g., `Vfoam [mL]` / height vs time)

2) Generates microscopy-like frames
- sample bubble radii from BD stats (mean area + variability)
- pack bubbles with non-overlap constraints
- render with noise/blur for microscopy-style appearance

3) Extracts bubble dynamics (OpenCV)
- segmentation + contours
- per-frame features: N(t), r_mean(t), r_std(t), circularity(t)
- saves overlays for visual verification

4) Links to stability
- computes half-life if the 50% threshold is observed in HD
- if not observed: reports “not reached within window” (stable during measurement window)
- computes a fast Lite stability score from coarsening trends (explainable)

5) Compares two runs
- Compare Two Runs panel: select Run A vs Run B
- shows coarsening rate, stability score, trend plots
- auto-label: “More stable”

## Repo structure
```
Graph-O-Foam/
  demo/                  # Streamlit app (demo/app.py)
  src/
    synth/               # XLSX → synthetic frame generator (generate_frames.py)
    tasks/               # Bubble dynamics extraction (extract_dynamics.py)
  data/
    uploads/             # Option B uploads (local only; gitignored)
    sheets/              # Option A local sheets (local only; gitignored)
    synth/               # Generated runs (local only; gitignored)
  assets/                # README images (tracked)
  requirements.txt
  README.md
  WRITEUP.md
```

## Demo flowchart
![Workflow](./assets/flowchart.png)

## Quickstart
### 1) Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 2) Launch the dashboard

```bash
PYTHONPATH=. streamlit run demo/app.py
```

Open: `http://localhost:8501`

---

## Follow-up Deliverables → `WRITEUP.md`

## Tools / Frameworks

Python 3.11, Streamlit, Pandas, OpenPyXL, NumPy, OpenCV (cv2), Git/GitHub.

---

## License

MIT

## Acknowledgments

- Thanks to entire TheChangeMakers team - "Atrerix"
- Inspired by Possibilities


[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()
