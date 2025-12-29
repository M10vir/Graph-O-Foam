<p align="center">
  <img src="assets/GoF_banner.jpeg" alt="Banner" width="100%" />
</p>

<h1 align="center">Graph-O-Foam — ActiveScan Copilot (Lite)</h1>
<h3 align="center">XLSX Datasheets → Synthetic Microscopy Frames → Bubble Dynamics → Graph Digital Twin → ML Stability Forecast</h3>

<p align="center">
  <b>Microscopy Hackathon 2025 — AISCIA Use Case (adapted for petroleum fluids)</b>
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

  <!-- Scikit-ML -->
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" />

  <!-- FFmpeg -->
  <img src="https://img.shields.io/badge/FFmpeg-Clip-007808?logo=ffmpeg&logoColor=white" />

  <!-- Git -->
  <img src="https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git&logoColor=white" />

  <!-- GitHub -->
  <img src="https://img.shields.io/badge/GitHub-Repo-181717?logo=github&logoColor=white" />

</p>

---

## 1) Problem Statement
Industrial petroleum fluids often produce unstable foams/emulsions where **bubble evolution** (coarsening, merging, deformation) determines performance and safety. In many real workflows, teams may not have labeled microscopy datasets — sometimes they only have **BD/HD datasheets (XLSX)**.

**Goal:** Build a fast, explainable, AI/ML-ready pipeline that transforms datasheets into microscopy-like frames, extracts bubble dynamics + graph-based digital twin metrics, and forecasts stability (half-life proxy), enabling GO vs NGO (or any condition A vs B) comparisons.

---

## 2) What Graph-O-Foam does (end-to-end)

### Phase-1 (Core pipeline)
1. **Ingest BD + HD XLSX**
   - **BD**: bubble statistics vs time (drives frame synthesis)
   - **HD**: foam stability signal (e.g., `Vfoam [mL]` / heights vs time)

2. **Generate synthetic microscopy frames**
   - samples bubble radii from BD stats
   - non-overlap placement / packing
   - renders microscopy-like images (noise/blur)

3. **Extract bubble dynamics (OpenCV)**
   - segmentation + contour analysis
   - per-frame: `n`, `r_mean`, `r_std`, `circularity`
   - saves verification overlays

4. **Stability estimate**
   - computes **half-life** when a 50% threshold is reached in the HD stability signal
   - if not reached: reports **“not reached within window”** (stable during the measured window)

### Phase-2 (Graph Digital Twin)
5. **Build bubble neighbor graph per frame**
   - adjacency via spatial proximity
   - exports `graph_metrics.csv` + overlay frames (`graph_overlays/`)

6. **Auto-create clip**
   - generates `graph_overlays.mp4` (+ optional GIF) from overlay frames

### Phase-2 (AI/ML compliance)
7. **ML model to forecast stability**
   - trains a baseline regressor (**RandomForest**) on extracted features across runs
   - uses early-window bubble + graph metrics to predict stability target (half-life proxy)

---

## 3) Outputs per run (in `data/synth/<run_name>/`)
- `frame_*.png` — synthetic microscopy frames  
- `bubble_dynamics.csv` — per-frame bubble dynamics  
- `overlays/` — bubble detection overlays  
- `graph_metrics.csv` — graph twin metrics  
- `graph_overlays/` — graph overlay frames  
- `graph_overlays.mp4` (+ optional `graph_overlays.gif`) — clip for quick demo  

> ✅ Phase-2 and ML use **only Phase-1 outputs** (`.png` + `.csv`) as source data.

---

## 4) Repo Structure
```text
Graph-O-Foam/
  demo/                     # Streamlit app (demo/app.py)
  src/
    synth/                  # XLSX → synthetic frame generator
    core/                   # bubble detection utilities
    tasks/                  # extract_dynamics, extract_graph_metrics, make_graph_overlay_clip
    ml/                     # ML training + inference
  data/
    uploads/                # Option B uploads (gitignored)
    sheets/                 # Option A local sheets (gitignored)
    synth/                  # generated runs (gitignored)
  assets/                   # README images (tracked)
  models/                   # trained ML model artifacts (tracked)
  requirements.txt
  README.md
  WRITEUP.md
```
---

```mermaid
flowchart TD
    A[Start] --> B{Input method?}

    B --> C1[Option A: Pick BD.xlsx + HD.xlsx from data/sheets]
    B --> C2[Option B: Upload BD.xlsx + HD.xlsx in Streamlit]

    C1 --> D[Read XLSX with pandas/openpyxl]
    C2 --> D

    D --> E[Phase-1: Synthetic frame generation]
    E --> F[Compute foam stability target from HD\n(Vfoam half-life if 50% reached)]
    F --> F1[Half-life computed (t at 50% Vfoam)]
    F --> F2[Half-life not reached → None/NA]

    E --> G[Sample bubble size stats from BD]
    G --> H[Place bubbles on canvas\n(non-overlap packing)]
    H --> I[Render microscopy-like frames\n(noise/blur/contrast)]
    I --> J[Outputs: frame_*.png + run folder]

    J --> K[Phase-1: Bubble dynamics extraction (OpenCV)]
    K --> L[Threshold + morphology]
    L --> M[Contours → bubble metrics per frame\n(n, r_mean, r_std, circularity...)]
    M --> N[Outputs: bubble_dynamics.csv + overlays/]

    N --> O[Phase-2: Bubble Neighbor Graph (Digital Twin)]
    O --> P[Build graph per frame\n(nodes=bubbles, edges=neighbors)]
    P --> Q[Compute graph metrics over time\n(avg_degree, density, giant_component...)]
    Q --> R[Outputs: graph_metrics.csv + graph_overlays/]

    R --> S[Clip Builder (ffmpeg)]
    S --> S1[graph_overlays.mp4]
    S --> S2[graph_overlays.gif]

    Q --> T[AI/ML Baseline (trained offline)]
    T --> U[Train RF regressor on early-window features\n(bubble + graph metrics)]
    U --> V[Saved model artifact\nmodels/*.joblib]

    N --> W[Streamlit Dashboard]
    R --> W
    S1 --> W
    S2 --> W
    V --> W

    W --> W1[Run viewer: frames, overlays, charts]
    W --> W2[Graph Twin viewer: metrics + overlays\nPlay clip if exists]
    W --> W3[Compare Run A vs Run B\n(winner = more stable)]
    W --> W4[One-click buttons\nGenerate graph metrics + clip\n(no CLI needed)]
```

## 5) Quickstart

### 5.1 Create environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Run Streamlit
```bash
PYTHONPATH=. streamlit run demo/app.py
```

Open: http://localhost:8501

---

## 6) Inputs: Option A vs Option B
- **Option A (Recommended for repeatability):** pick BD/HD from `data/sheets/`
- **Option B (Upload in UI):** upload BD/HD XLSX in Streamlit  
  - run folder names are **sanitized** to avoid filesystem/FFmpeg issues (e.g., `[]`, spaces, etc.)

---

## 7) Train the ML baseline (optional but recommended)
```bash
PYTHONPATH=. python src/ml/train_model.py
```

Model artifact:
- `models/coarsening_rf.joblib`

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
- Microscopy Hackathon 2025 — AISCIA Use Case 


[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()
