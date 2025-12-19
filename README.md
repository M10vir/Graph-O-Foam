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
```text
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

## Demo flowchart
![Workflow](assets/flowchart.png)

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

## Using the dashboard

### Option A — Select BD/HD from a folder (recommended for many sheets)

1. Put your XLSX files under a local folder (example): `data/sheets/`
2. In the dashboard sidebar: choose Option A, then select your BD and HD
3. Click Generate / Run

### Option B — Upload BD/HD in the dashboard (demo-friendly)

1. In the dashboard sidebar: choose Option B
2. Upload BD.xlsx and HD.xlsx
3. Click Generate / Run

---

## Outputs (per run)

Each run creates a folder like `data/synth/<run_name>/` containing:

* `frame_*.png` — synthetic microscopy-like frames
* `overlays/` — verification overlays (segmentation/contours)
* `bubble_dynamics.csv` — extracted dynamics (N, r_mean, r_std, circularity vs time)
* `frames_metadata.csv` — run metadata including stability fields

---

## Compare Two Runs 

Generate and extract dynamics for at least two runs, then scroll to:

🔁 Compare Two Runs (GO vs NGO / Condition A vs B)

You’ll see:

* coarsening rate (Δr/Δt)
* stability score (Lite)
* aligned plots: N(t), r_mean(t), circularity(t)
* automatic “More stable” label

---

## CLI (optional)

If you want to run steps manually:

### Generate frames

```bash
PYTHONPATH=. python src/synth/generate_frames.py \
  --bd "path/to/BD.xlsx" \
  --hd "path/to/HD.xlsx" \
  --out data/synth/run1 \
  --nframes 40
```

### Extract dynamics

```bash
PYTHONPATH=. python src/tasks/extract_dynamics.py --folder data/synth/run1
```

---

## Troubleshooting

### “ModuleNotFoundError: No module named 'src'”

Run Streamlit with:

```bash
PYTHONPATH=. streamlit run demo/app.py
```

### Half-life shows N/A / not reached

That’s expected if HD never crosses the 50% threshold within the measurement window. We intentionally report this honestly as not reached within window.

---

## Submission Deliverables

* ✅ Code: GitHub repo (this repository)
* ✅ Write-up: `WRITEUP.md`
* ✅ Demo video: 2-minute screen recording (dashboard walkthrough)

### 2-minute demo checklist

1. Launch Streamlit
2. Generate Run A (BD+HD) → Extract dynamics
3. Generate Run B (BD+HD) → Extract dynamics
4. Show Compare Two Runs (scores + winner label + plots)
5. End with takeaway: “XLSX → frames → dynamics → stability + comparison”

---

## Tools / Frameworks

Python 3.11, Streamlit, Pandas, OpenPyXL, NumPy, OpenCV (cv2), Git/GitHub.

---

## License

MIT (or update as needed)

```
::contentReference[oaicite:0]{index=0}
```

