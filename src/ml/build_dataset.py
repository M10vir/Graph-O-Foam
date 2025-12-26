from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

DATA_SYNTH = Path("data/synth")
OUT_CSV = Path("data/ml/run_level_features.csv")

def run_family(name: str) -> str:
    # Remove trailing _HHMMSS or _###### patterns to avoid leakage
    return re.sub(r"[_-]\d{6,}$", "", name)

def slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]):
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])

def time_to_threshold(t: np.ndarray, series: np.ndarray, thr: float) -> float | None:
    idx = np.where(series >= thr)[0]
    if len(idx) == 0:
        return None
    return float(t[int(idx[0])])

def build_for_run(run_dir: Path, early_frac: float = 0.3) -> dict | None:
    bd = run_dir / "bubble_dynamics.csv"
    gm = run_dir / "graph_metrics.csv"
    if not bd.exists() or not gm.exists():
        return None

    bdf = pd.read_csv(bd).dropna(subset=["t_s"])
    gdf = pd.read_csv(gm).dropna(subset=["t_s"])

    # Merge on time (nearest merge to tolerate slightly different sampling)
    bdf = bdf.sort_values("t_s")
    gdf = gdf.sort_values("t_s")
    df = pd.merge_asof(bdf, gdf, on="t_s", direction="nearest")

    if len(df) < 8:
        return None

    n = len(df)
    k = max(5, int(n * early_frac))
    early = df.iloc[:k]
    full = df

    tE = early["t_s"].to_numpy()
    tF = full["t_s"].to_numpy()

    rE = early["r_mean"].to_numpy()
    rF = full["r_mean"].to_numpy()

    # Targets computed from full run (labels)
    coarsen_rate = slope(tF, rF)
    r0 = float(rF[0])
    thr = 1.5 * r0
    t_to_15x = time_to_threshold(tF, rF, thr)  # proxy half-life
    # If not reached, keep as NaN (censored-ish)
    t_to_15x = float("nan") if t_to_15x is None else t_to_15x

    feats = {
        "run": run_dir.name,
        "family": run_family(run_dir.name),
        "n_frames": n,

        # Foam features (early-window)
        "r_mean_early_mean": float(np.nanmean(early["r_mean"])),
        "r_mean_early_std": float(np.nanstd(early["r_mean"])),
        "r_mean_early_slope": slope(tE, rE),
        "n_early_mean": float(np.nanmean(early["n"])),
        "n_early_slope": slope(tE, early["n"].to_numpy()),
        "circ_early_mean": float(np.nanmean(early["circ_mean"])),

        # Graph/physics proxy features (early-window)
        "avg_degree_early_mean": float(np.nanmean(early.get("avg_degree", np.nan))),
        "avg_degree_early_slope": slope(tE, early.get("avg_degree", pd.Series(np.nan)).to_numpy()),
        "density_early_mean": float(np.nanmean(early.get("density", np.nan))),
        "density_early_slope": slope(tE, early.get("density", pd.Series(np.nan)).to_numpy()),
        "gcr_early_mean": float(np.nanmean(early.get("giant_component_ratio", np.nan))),
        "gcr_early_slope": slope(tE, early.get("giant_component_ratio", pd.Series(np.nan)).to_numpy()),

        # Optional energy proxy (dimensionless, explainable)
        "energy_proxy_early": float(np.nanmean(early["n"] * (early["r_mean"] ** 2))),

        # Labels/targets
        "y_coarsen_rate": coarsen_rate,
        "y_t_to_15x_rmean": t_to_15x,
    }
    return feats

def main():
    rows = []
    for d in sorted(DATA_SYNTH.iterdir()):
        if d.is_dir():
            r = build_for_run(d)
            if r:
                rows.append(r)

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote dataset: {OUT_CSV}  (rows={len(out)})")
    print(out.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
