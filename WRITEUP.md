````md
# Graph-O-Foam: ActiveScan Copilot (Lite)
## Machine Learning for Microscopy Hackathon 2025 — AISCIA Use Case (Doha, Qatar)

### 1) Executive Summary
Foam stability depends on how bubbles evolve over time (coarsening, merging, deformation). A key challenge in microscopy-driven workflows is converting raw observations into **quantitative bubble dynamics** and a **clear stability comparison** between formulations (e.g., GO vs NGO).  
**Graph-O-Foam: ActiveScan Copilot (Lite)** delivers an explainable pipeline that goes from **BD + HD datasheets (XLSX)** to **microscopy-like frames**, extracts bubble dynamics using **computer vision**, and provides a **stability forecast + run-to-run comparison** inside an interactive dashboard.

---

### 2) Problem Statement
Microscopy images contain rich information about bubble size distributions and coarsening behavior, but:
- Manual interpretation is slow and inconsistent.
- Experiments often produce heterogeneous data: sometimes only datasheets (BD/HD), sometimes images, sometimes both.
- Teams need a repeatable way to **quantify bubble patterns**, **track dynamics over time**, and **compare stability across conditions**.

This hackathon use case asks for workflows that link bubble patterns in microscopy images to foam stability metrics such as coarsening rate, bubble size distribution shifts, and foam half-life.

---

### 3) Data and Inputs
We operate on paired spreadsheets:
- **BD (Bubble Dynamics sheet, XLSX):** time-series bubble statistics (e.g., bubble count density, mean area, std area).
- **HD (Foam stability sheet, XLSX):** time-series foam stability signal (e.g., `Vfoam [mL]`, foam height/volume vs time), enabling half-life estimation when the 50% threshold is observed.

The app supports:
- **Option A:** select BD/HD from a local folder (batch-friendly)
- **Option B:** upload BD/HD directly in the dashboard (demo-friendly)

---

### 4) Workflow Overview
**End-to-end flow:**  
**XLSX datasheets (BD+HD) → synthetic microscopy frames → bubble dynamics → stability forecast (half-life + score) → GO vs NGO comparison**

#### 4.1 Synthetic microscopy frame generation (from BD)
When raw microscopy images are unavailable, the pipeline generates microscopy-like frames:
- bubble radii are sampled from BD-derived distributions (mean area + variability)
- bubbles are placed with non-overlap constraints
- frames are rendered with noise/blur to resemble microscopy appearance  
**Output:** a sequence of `frame_*.png` plus `frames_metadata.csv`

#### 4.2 Bubble detection and dynamics extraction (OpenCV)
For each generated frame, we extract explainable features using computer vision:
- thresholding + morphological operations
- contour detection and filtering
- per-frame metrics:
  - **N(t):** bubble count
  - **r_mean(t):** mean bubble radius
  - **r_std(t):** radius variability
  - **circularity(t):** shape stability indicator  
**Output:** `bubble_dynamics.csv` and `overlays/` (visual verification of detection)

#### 4.3 Stability metrics (HD + dynamics)
- **Half-life (observed):** first time `Vfoam` drops below 50% of the initial value, with interpolation when applicable.
- **Not reached within window:** if the dataset never crosses the 50% threshold, the method reports “not reached within window” (stable during measurement).
- **Lite stability score:** a fast, explainable stability estimate derived from coarsening trends (e.g., slope of `r_mean(t)`), used for ranking and comparison.

#### 4.4 Compare conditions (GO vs NGO / Run A vs Run B)
The Streamlit dashboard includes a **Compare Two Runs** panel:
- select Run A and Run B
- show coarsening rate (Δr/Δt), stability scores
- show aligned trend plots (N(t), r_mean(t), circularity)
- auto-label which run appears **more stable** based on the score

---

### 5) Results and Key Insights
**What the workflow produces:**
- A reproducible run folder with:
  - synthetic frames (`frame_*.png`)
  - overlays (segmentation/contours for verification)
  - `bubble_dynamics.csv` with time-series bubble metrics
  - `frames_metadata.csv` including stability fields

**Insights we extract:**
- **Coarsening behavior:** increasing `r_mean(t)` trends indicate coarsening progression.
- **Distribution changes:** `r_std(t)` highlights widening/narrowing of size distributions.
- **Shape stability:** decreasing circularity can indicate deformation/merging events.
- **Stability interpretation:** combining HD half-life (when observed) with dynamics-based scoring supports fast ranking of conditions.

---

### 6) Why this approach is hackathon-strong
- **Works even without microscopy images:** datasheets → frames → measurable dynamics.
- **Explainable end-to-end:** every plot is backed by detection overlays and transparent computations.
- **Comparison-ready:** GO vs NGO (or any two conditions) becomes a one-click, judge-friendly decision.
- **Reusable workflow:** new XLSX pairs instantly produce a new run and can be compared.

---

### 7) Reproducibility (How to run)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. streamlit run demo/app.py
````

**Expected outputs per run:**

* `data/synth/<run>/frame_*.png`
* `data/synth/<run>/overlays/`
* `data/synth/<run>/bubble_dynamics.csv`
* `data/synth/<run>/frames_metadata.csv`

---

### 8) Limitations and Next Steps

**Limitations**

* Synthetic frames approximate microscopy appearance; full validation should include real microscopy frames when available.
* Half-life may not be observed in some datasets; we report “not reached within window” rather than forcing a prediction.

**Next steps**

* Add richer distribution features (quantiles, skewness) and event markers (merge/split likelihood).
* Train a supervised model once enough labeled conditions exist (e.g., stability labels/half-life across many formulations).
* Extend to direct ingestion of real microscopy images when provided.

---

### 9) Conclusion

Graph-O-Foam demonstrates a practical, explainable workflow for foam stability analysis in microscopy contexts. By transforming datasheets into frames, extracting interpretable bubble dynamics, and enabling run-to-run comparisons, the project supports rapid formulation screening and clear stability insights aligned with the hackathon’s bubble dynamics and foam stability objectives.

```
::contentReference[oaicite:0]{index=0}
```

