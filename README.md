````md
## Problem
Foams evolve fast: bubbles merge, grow (coarsening), and reshape over time. In the hackathon use case, we need an ML workflow that can **quantify bubble dynamics** and **link image patterns to foam stability (half-life)**, enabling **GO vs NGO** comparisons.

## Solution — Graph-O-Foam: ActiveScan Copilot (Lite)
A fast, explainable pipeline that goes:
**XLSX datasheets (BD + HD) → synthetic microscopy frames → bubble dynamics → stability forecast + comparison**

Even if you start with **datasheets only**, the system generates microscopy-like frames and extracts bubble metrics over time.

## How it works
1. **Ingest XLSX**
   - **BD sheet**: bubble field statistics over time (drives frame synthesis)
   - **HD sheet**: foam stability signal (half-life if reached; otherwise “not reached within window”)
2. **Synthetic microscopy frame generation**
   - sample bubble radii from BD stats
   - place bubbles on a canvas (non-overlap heuristic)
   - render frames with noise/blur for microscopy-like appearance
3. **Bubble dynamics extraction (OpenCV)**
   - segmentation + contours
   - metrics per frame: **N(t), r_mean(t), r_std(t), circularity(t)**
   - overlays for visual verification
4. **Stability forecast (Lite)**
   - coarsening rate from **r_mean(t)** + distribution/shape trends → an interpretable stability score
5. **Dashboard (Streamlit)**
   - time slider, frames + overlays, plots, exports
   - run multiple conditions to compare **GO vs NGO** (or any two datasets)

## Tools / Frameworks
Python 3.11, Streamlit, Pandas, OpenPyXL, NumPy, OpenCV (cv2), Git/GitHub.

## Demo
### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

### Option A — Select XLSX from folder

```bash
mkdir -p data/sheets
# copy your BD/HD xlsx into data/sheets/
PYTHONPATH=. streamlit run demo/app.py
```

In the sidebar: pick BD + HD → **Generate frames + extract dynamics** → explore the run.

### Option B — Upload XLSX from dashboard

```bash
PYTHONPATH=. streamlit run demo/app.py
```

In the sidebar: choose **Option B** → upload BD + HD → **Generate**.

## Outputs

Each run creates:

* `data/synth/<run>/frame_*.png`
* `data/synth/<run>/overlays/`
* `data/synth/<run>/bubble_dynamics.csv`
* `data/synth/<run>/frames_metadata.csv` (includes half-life + method label when available)

## Why it’s hackathon-strong

* **Visual + measurable**: see bubbles, see metrics, see stability trend.
* **Works even without raw microscopy**: datasheets → frames → dynamics.
* **Explainable**: CV features + clear stability logic, easy to defend to judges.

```
::contentReference[oaicite:0]{index=0}
```

