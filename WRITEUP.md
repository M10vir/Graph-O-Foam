# Graph-O-Foam — ActiveScan Copilot (Phase 1 + Phase 2)
## Machine Learning for Microscopy Hackathon 2025 — AISCIA Use Case (Doha, Qatar)

### 1) Executive Summary
Foam/emulsion stability in **petroleum liquids** depends on how bubbles evolve over time (coarsening, merging, deformation) and how quickly the bulk foam volume decays. The practical challenge is that teams often have **datasheets (BD + HD XLSX)** but not consistent microscopy imagery — and manual interpretation is slow.

**Graph-O-Foam** delivers an end-to-end, **explainable** pipeline:
**BD+HD datasheets → synthetic microscopy frames → bubble dynamics → graph-based digital twin metrics → ML stability forecast + GO vs NGO comparison**, all inside a Streamlit dashboard.

This project is designed to be **hackathon-ready**: reproducible, demo-friendly, and grounded in measurable outputs (`.png`, `.csv`, `.mp4/.gif`).

---

### 2) Problem Statement
Microscopy contains rich signals about bubble size distributions and coarsening behavior, but:
- manual inspection is slow and inconsistent,
- datasets are heterogeneous (sometimes only datasheets, sometimes images),
- stakeholders need a repeatable way to **quantify bubble patterns**, track dynamics, and **compare stability across formulations/conditions** (e.g., GO vs NGO).

The hackathon asks for workflows linking bubble patterns to foam stability metrics (coarsening rate, distribution shift, half-life).

---

### 3) Inputs and Data Sources
We operate on **paired spreadsheets**:

- **BD (Bubble Dynamics, XLSX)**: bubble statistics over time (e.g., mean area / variability, count density, etc.).
- **HD (Foam Stability / Height-Volume, XLSX)**: foam height/volume vs time (e.g., `Vfoam [mL]`) used to estimate **half-life** (time to 50% of initial foam volume).

**Dashboard modes**
- **Option A (batch friendly)**: pick BD/HD from a local folder.
- **Option B (demo friendly)**: upload BD/HD directly in Streamlit (run names are sanitized to avoid special-character path issues).

---

### 4) Workflow Overview (Current Implementation)
**End-to-end flow:**  
**XLSX datasheets (BD+HD) → synthetic microscopy frames → bubble dynamics → graph digital twin → stability forecast (half-life + ML) → GO vs NGO comparison**

#### 4.1 Synthetic microscopy frame generation (from BD)
When raw microscopy images are unavailable, we generate microscopy-like frames:
- bubble radii are sampled from BD-derived distributions (mean + variability)
- bubbles placed with non-overlap constraints
- noise/blur injected to resemble microscopy texture

**Output (per run):**
- `frame_*.png`
- `frames_metadata.csv` (includes time index and stability fields)

#### 4.2 Bubble detection & dynamics extraction (OpenCV)
For each frame we extract interpretable features:
- thresholding + morphological ops
- contour detection + filtering
- per-frame metrics:
  - **N(t)** bubble count
  - **r_mean(t)** mean radius
  - **r_std(t)** radius variability
  - **circularity(t)** shape stability indicator

**Output (per run):**
- `bubble_dynamics.csv`
- `overlays/` (visual verification)

#### 4.3 Half-life estimation (from HD)
We compute observed half-life as:
- first time `Vfoam` drops below **50%** of its initial value (linear interpolation when applicable)
- if the 50% threshold is **not reached**, we report “not reached within window” (stable during measurement) rather than forcing a number

**Output:** half-life is written into metadata and displayed in the dashboard.

#### 4.4 Phase-2: Bubble Neighbor Graph (Digital Twin)
From detected bubbles at each time step, we build a **neighbor graph**:
- nodes = bubbles
- edges = proximity-based neighbors (distance threshold / k-NN style link)
- compute graph metrics per frame such as:
  - average degree
  - density
  - giant component ratio
  - simple energy proxy (physics-inspired, based on connectivity + bubble size scale)

**Output (per run):**
- `graph_metrics.csv`
- `graph_overlays/` (edges overlaid on frames)
- optional clip: `graph_overlays.mp4` and/or `graph_overlays.gif`

#### 4.5 AI/ML: Baseline stability predictor (Random Forest)
To satisfy the AI/ML component **without overclaiming**, we implemented a fast baseline:
- derive **early-window features** from dynamics + graph metrics (early mean + slope)
- train a **RandomForest regressor** (scikit-learn) across runs
- save model artifact: `models/coarsening_rf.joblib`
- report evaluation (MAE/RMSE/R²) and feature importances for explainability

This gives a practical stability forecast (predictive signal) even when half-life is not observed within the HD window.

---

### 5) Results and Key Insights (What we can show to judges)
Each run creates a reproducible folder with:
- synthetic microscopy frames (`frame_*.png`)
- bubble overlays (`overlays/`)
- bubble dynamics time series (`bubble_dynamics.csv`)
- graph digital twin metrics (`graph_metrics.csv`)
- graph overlays (`graph_overlays/`)
- optional playable clip (`graph_overlays.mp4` / `graph_overlays.gif`)
- optional ML forecast (dashboard + model artifact)

**Interpretation highlights**
- increasing `r_mean(t)` indicates coarsening progression
- `r_std(t)` captures distribution widening/narrowing
- circularity trends capture deformation/merging signatures
- graph metrics (degree / giant component) provide a **digital twin** view of bubble neighborhood evolution
- ML feature importances provide quick insight into which early signals correlate most with stability outcomes

---

### 6) Demo Experience (Streamlit)
The dashboard supports:
- generate runs (Option A or Option B)
- inspect frames and overlays
- compare two runs (GO vs NGO / Run A vs Run B)
- Phase-2: show graph digital twin metrics and overlays
- one-click utilities (when enabled): generate missing dynamics/graph metrics/clip for selected run
- ML: show predicted stability signal based on trained baseline model

---

### 7) Reproducibility (How to run)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# (Recommended) Install ffmpeg for mp4/gif clip creation
# macOS:
brew install ffmpeg

PYTHONPATH=. streamlit run demo/app.py
```

**Optional: Train baseline ML model**
```bash
PYTHONPATH=. python src/ml/train_model.py
```

---

### 8) Limitations and Next Steps
**Current limitations**
- synthetic frames approximate microscopy; validation with real microscopy is the next step
- half-life may not be observed in short windows; we report “not reached within window”
- current ML is a **baseline** (Random Forest) trained on limited runs

**Next (Phase-3 / expansion)**
- ingest real microscopy frames when available (same downstream pipeline)
- richer physics-informed features (e.g., energy-based potentials from graph evolution)
- stronger temporal models (sequence models) once labeled data volume grows
- automated batch processing across folders and reporting (summary PDF/CSV export)

---

### 9) Conclusion
Graph-O-Foam demonstrates an **explainable**, end-to-end approach for foam/emulsion stability analysis in petroleum liquids. By transforming datasheets into microscopy-like frames, extracting bubble dynamics, building a graph digital twin, and adding an ML baseline forecast, the project provides a judge-friendly workflow for rapid formulation screening and GO vs NGO comparisons.
