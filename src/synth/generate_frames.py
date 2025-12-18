from pathlib import Path
import numpy as np
import pandas as pd

def load_sheet(xlsx_path: str):
    # reads first sheet by default
    return pd.read_excel(xlsx_path)

def _pick_col(df: pd.DataFrame, candidates: list[str], contains_any: list[str] | None = None):
    # 1) exact matches first
    for c in candidates:
        if c in df.columns:
            return c

    # 2) case-insensitive exact match
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        k = str(c).strip().lower()
        if k in lower_map:
            return lower_map[k]

    # 3) "contains" heuristic
    if contains_any:
        cols = [str(c) for c in df.columns]
        cols_lower = [c.lower() for c in cols]
        for i, cl in enumerate(cols_lower):
            if all(tok.lower() in cl for tok in contains_any):
                return df.columns[i]

    return None

def _half_life_observed(t: np.ndarray, v: np.ndarray) -> float | None:
    """
    Observed half-life: first time when v <= 0.5*v0, with linear interpolation.
    """
    if len(t) < 2:
        return None
    v0 = float(v[0])
    if not np.isfinite(v0) or v0 <= 0:
        return None

    target = 0.5 * v0
    idx = np.where(v <= target)[0]
    if len(idx) == 0:
        return None

    i = int(idx[0])
    if i == 0:
        return float(t[0])

    t1, t2 = float(t[i - 1]), float(t[i])
    v1, v2 = float(v[i - 1]), float(v[i])
    if v2 == v1:
        return float(t2)

    frac = (target - v1) / (v2 - v1)
    return float(t1 + frac * (t2 - t1))

def _half_life_extrapolated(t: np.ndarray, v: np.ndarray) -> float | None:
    """
    Extrapolated half-life using exponential decay fit when half-life is not reached:
      ln(v) = a + b*t   (requires b < 0)
    Returns t_half (seconds) or None if trend isn't decaying.
    """
    if len(t) < 5:
        return None

    v0 = float(v[0])
    if not np.isfinite(v0) or v0 <= 0:
        return None

    target = 0.5 * v0

    # Need positive values for log
    mask = np.isfinite(t) & np.isfinite(v) & (v > 0)
    t2, v2 = t[mask], v[mask]
    if len(t2) < 5:
        return None

    # If there is basically no decay, don't extrapolate
    if float(np.min(v2)) > 0.95 * v0:
        return None

    y = np.log(v2)
    b, a = np.polyfit(t2, y, 1)  # y = b*t + a
    if not np.isfinite(b) or b >= 0:
        return None

    t_half = (np.log(target) - a) / b
    if not np.isfinite(t_half):
        return None

    # Extrapolated half-life should be AFTER the last observed time if not reached
    if t_half <= float(np.max(t2)):
        return None

    return float(t_half)

def foam_half_life_seconds(df_hd: pd.DataFrame, extrapolate: bool = True):
    """
    Returns:
      best_s: float | None
      observed_s: float | None
      extrapolated_s: float | None
      method: "observed" | "extrapolated" | "not_reached" | "missing"
    """
    # Prefer your original expected names first
    t_candidates = ["t [s]", "t[s]", "time", "time_s", "Time", "Time (s)", "t"]
    v_candidates = ["Vfoam [mL]", "Vfoam", "foam_volume", "Foam Volume", "volume", "V", "V [mL]",
                    "hfoam [mm]", "Hfoam [mm]", "Hfoam", "height", "H", "H [mm]"]

    tcol = _pick_col(df_hd, t_candidates, contains_any=["t"])
    vcol = _pick_col(df_hd, v_candidates, contains_any=["foam"])

    # Fallback: first 2 numeric columns
    if tcol is None or vcol is None:
        num_cols = [c for c in df_hd.columns if pd.api.types.is_numeric_dtype(df_hd[c])]
        if len(num_cols) >= 2:
            tcol = tcol or num_cols[0]
            vcol = vcol or num_cols[1]

    if tcol is None or vcol is None:
        return None, None, None, "missing"

    t = pd.to_numeric(df_hd[tcol], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df_hd[vcol], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(t) & np.isfinite(v)
    t, v = t[mask], v[mask]
    if len(t) < 2:
        return None, None, None, "missing"

    # Sort by time
    order = np.argsort(t)
    t, v = t[order], v[order]

    observed = _half_life_observed(t, v)
    if observed is not None:
        return observed, observed, None, "observed"

    if extrapolate:
        est = _half_life_extrapolated(t, v)
        if est is not None:
            return est, None, est, "extrapolated"

    return None, None, None, "not_reached"

def sample_radii_from_area(mean_area, std_area, n, rng):
    mean_area = max(float(mean_area), 1e-6)
    std_area = max(float(std_area), 1e-6)
    sigma2 = np.log(1 + (std_area**2) / (mean_area**2))
    mu = np.log(mean_area) - 0.5 * sigma2
    sigma = np.sqrt(sigma2)
    areas = rng.lognormal(mean=mu, sigma=sigma, size=int(n))
    r_um = np.sqrt(areas / np.pi)
    return r_um

def place_circles(canvas_px, radii_px, rng, max_tries=20000):
    H, W = canvas_px
    centers = []
    tries = 0
    for r in radii_px:
        placed = False
        while tries < max_tries and not placed:
            tries += 1
            x = int(rng.integers(r, W - r))
            y = int(rng.integers(r, H - r))
            ok = True
            for (cx, cy, cr) in centers:
                if (x - cx) ** 2 + (y - cy) ** 2 < (0.85 * (r + cr)) ** 2:
                    ok = False
                    break
            if ok:
                centers.append((x, y, r))
                placed = True
        if not placed:
            break
    return centers

def render_frame(centers, canvas_px=(512, 512), noise=0.06, blur=1):
    import cv2
    H, W = canvas_px
    img = np.zeros((H, W), dtype=np.float32)

    for x, y, r in centers:
        cv2.circle(img, (x, y), int(r), 0.75, thickness=-1)
        cv2.circle(img, (x, y), int(r), 1.0, thickness=1)

    yy, xx = np.mgrid[0:H, 0:W]
    grad = (0.10 + 0.10 * (yy / H) + 0.06 * (xx / W)).astype(np.float32)
    img = np.clip(img + grad, 0, 1)

    if blur and blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)

    img = np.clip(img + np.random.normal(0, noise, size=img.shape).astype(np.float32), 0, 1)
    return (img * 255).astype(np.uint8)

def generate_sequence(bd_xlsx, hd_xlsx, out_dir, n_frames=40, seed=7, extrapolate_half_life=True):
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_bd = load_sheet(bd_xlsx)
    df_hd = load_sheet(hd_xlsx)

    half_best, half_obs, half_est, hl_method = foam_half_life_seconds(df_hd, extrapolate=extrapolate_half_life)

    # BD time column
    bd_tcol = _pick_col(df_bd, ["t [s]", "t[s]", "time", "Time", "Time (s)", "t"], contains_any=["t"])
    if bd_tcol is None:
        raise ValueError(f"BD sheet missing time column. Found columns: {list(df_bd.columns)}")

    df_bd = df_bd.dropna(subset=[bd_tcol]).copy()
    df_bd[bd_tcol] = pd.to_numeric(df_bd[bd_tcol], errors="coerce")
    df_bd = df_bd.dropna(subset=[bd_tcol]).copy()
    df_bd = df_bd.sort_values(bd_tcol)

    if n_frames <= 0 or n_frames >= len(df_bd):
        df_sel = df_bd.reset_index(drop=True)
    else:
        idxs = np.linspace(0, len(df_bd) - 1, int(n_frames)).astype(int)
        df_sel = df_bd.iloc[idxs].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    px_per_um = 512 / 1000.0

    bc_col = _pick_col(df_bd, ["BC [mm⁻²]", "BC [mm^-2]", "BC"], contains_any=["bc"])
    mean_area_col = _pick_col(df_bd, ["M̅B̅A̅ [µm²]", "MBA [µm²]", "Mean Bubble Area"], contains_any=["mba"])
    std_area_col  = _pick_col(df_bd, ["SD M̅B̅A̅ [µm²]", "SD MBA [µm²]", "Std Bubble Area"], contains_any=["sd", "mba"])

    rows = []
    for i, row in df_sel.iterrows():
        t = float(row[bd_tcol])

        bc = float(row.get(bc_col, np.nan)) if bc_col else np.nan
        mean_area = float(row.get(mean_area_col, np.nan)) if mean_area_col else np.nan
        std_area = float(row.get(std_area_col, np.nan)) if std_area_col else np.nan

        if np.isfinite(bc):
            n_bubbles = int(np.clip(bc, 60, 450))
        else:
            n_bubbles = 200

        if not np.isfinite(mean_area):
            mean_area = 2500.0
        if not np.isfinite(std_area):
            std_area = 800.0

        r_um = sample_radii_from_area(mean_area, std_area, n_bubbles, rng)
        r_px = np.clip(r_um * px_per_um, 3, 80)

        centers = place_circles((512, 512), r_px, rng)
        img = render_frame(centers, canvas_px=(512, 512))

        fname = f"frame_{i:03d}_t{int(t)}.png"
        (out_dir / fname).write_bytes(cv2.imencode(".png", img)[1].tobytes())

        rows.append({
            "frame": fname,
            "t_s": t,

            # best = observed if available else extrapolated else None
            "half_life_s": half_best,
            "half_life_observed_s": half_obs,
            "half_life_extrapolated_s": half_est,
            "half_life_method": hl_method,

            "n_bubbles_rendered": len(centers),
            "mean_area_um2": float(mean_area),
            "std_area_um2": float(std_area),
            "bc_mm2": float(bc) if np.isfinite(bc) else np.nan,
        })

    pd.DataFrame(rows).to_csv(out_dir / "frames_metadata.csv", index=False)
    return half_best, hl_method

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bd", required=True, help="BD.xlsx path")
    ap.add_argument("--hd", required=True, help="HD.xlsx path")
    ap.add_argument("--out", default="data/synth/run1", help="output folder")
    ap.add_argument("--nframes", type=int, default=40)
    ap.add_argument("--no-extrapolate", action="store_true", help="Disable half-life extrapolation")
    args = ap.parse_args()

    hl, method = generate_sequence(
        args.bd, args.hd, args.out, args.nframes,
        extrapolate_half_life=(not args.no_extrapolate)
    )
    print("✅ Generated synthetic frames in:", args.out)
    print("Foam half-life (s):", hl, "| method:", method)
