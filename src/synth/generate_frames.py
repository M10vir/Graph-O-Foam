from pathlib import Path
import numpy as np
import pandas as pd

def load_sheet(xlsx_path: str):
    # reads first sheet by default; your files each have a single sheet
    return pd.read_excel(xlsx_path)

def foam_half_life_seconds(df_hd: pd.DataFrame):
    # expects columns like: t [s], Vfoam [mL]
    t = df_hd["t [s]"].astype(float).values
    v = df_hd["Vfoam [mL]"].astype(float).values
    v0 = v[0]
    target = 0.5 * v0
    # find first time crossing below target
    idx = np.where(v <= target)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return float(t[0])
    # linear interpolation
    t1, t2 = t[i-1], t[i]
    v1, v2 = v[i-1], v[i]
    if v2 == v1:
        return float(t2)
    frac = (target - v1) / (v2 - v1)
    return float(t1 + frac * (t2 - t1))

def sample_radii_from_area(mean_area, std_area, n, rng):
    # area in µm^2; sample positive areas, convert to radius (µm)
    # use lognormal proxy to keep areas positive
    mean_area = max(mean_area, 1e-6)
    std_area = max(std_area, 1e-6)
    sigma2 = np.log(1 + (std_area**2)/(mean_area**2))
    mu = np.log(mean_area) - 0.5*sigma2
    sigma = np.sqrt(sigma2)
    areas = rng.lognormal(mean=mu, sigma=sigma, size=n)
    r_um = np.sqrt(areas / np.pi)
    return r_um

def place_circles(canvas_px, radii_px, rng, max_tries=20000):
    # simple rejection sampling placement
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
                # allow mild overlap (0.85 factor)
                if (x - cx)**2 + (y - cy)**2 < (0.85*(r + cr))**2:
                    ok = False
                    break
            if ok:
                centers.append((x, y, r))
                placed = True
        # if we can’t place all, stop early
        if not placed:
            break
    return centers

def render_frame(centers, canvas_px=(512,512), noise=0.06, blur=1):
    import cv2
    H, W = canvas_px
    img = np.zeros((H, W), dtype=np.float32)

    # draw filled bubbles (slightly brighter interiors)
    for x, y, r in centers:
        cv2.circle(img, (x, y), int(r), 0.75, thickness=-1)
        cv2.circle(img, (x, y), int(r), 1.0, thickness=1)

    # background gradient (microscopy vibe)
    yy, xx = np.mgrid[0:H, 0:W]
    grad = (0.10 + 0.10*(yy/H) + 0.06*(xx/W)).astype(np.float32)
    img = np.clip(img + grad, 0, 1)

    if blur and blur > 0:
        img = cv2.GaussianBlur(img, (0,0), blur)

    # noise
    img = np.clip(img + np.random.normal(0, noise, size=img.shape).astype(np.float32), 0, 1)
    return (img * 255).astype(np.uint8)

def generate_sequence(bd_xlsx, hd_xlsx, out_dir, n_frames=40, seed=7):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_bd = load_sheet(bd_xlsx)
    df_hd = load_sheet(hd_xlsx)

    # Half-life from height/volume sheet
    half_life = foam_half_life_seconds(df_hd)

    # Pick evenly spaced timepoints from BD sheet
    df_bd = df_bd.dropna(subset=["t [s]"]).copy()
    df_bd["t [s]"] = df_bd["t [s]"].astype(float)
    times = df_bd["t [s]"].values
    if n_frames <= 0 or n_frames >= len(df_bd):
        df_sel = df_bd.reset_index(drop=True)   # ALL rows/timepoints
    else:
        idxs = np.linspace(0, len(df_bd)-1, n_frames).astype(int)
        df_sel = df_bd.iloc[idxs].reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # Scale: assume 1 mm field of view mapped to 512 px → 1 µm ≈ 0.512 px
    px_per_um = 512 / 1000.0

    rows = []
    for i, row in df_sel.iterrows():
        t = float(row["t [s]"])
        bc = float(row.get("BC [mm⁻²]", np.nan))
        mean_area = float(row.get("M̅B̅A̅ [µm²]", row.get("MBA [µm²]", np.nan)))
        std_area = float(row.get("SD M̅B̅A̅ [µm²]", row.get("SD MBA [µm²]", np.nan)))

        # Decide number of bubbles:
        # If BC exists, interpret as bubbles per mm^2. Use 1 mm^2 FOV.
        # Clamp to reasonable for rendering speed.
        if np.isfinite(bc):
            n_bubbles = int(np.clip(bc, 60, 450))
        else:
            n_bubbles = 200

        r_um = sample_radii_from_area(mean_area, std_area, n_bubbles, rng)
        r_px = np.clip(r_um * px_per_um, 3, 80)

        centers = place_circles((512,512), r_px, rng)
        img = render_frame(centers, canvas_px=(512,512))

        fname = f"frame_{i:03d}_t{int(t)}.png"
        (out_dir / fname).write_bytes(__import__("cv2").imencode(".png", img)[1].tobytes())

        rows.append({
            "frame": fname,
            "t_s": t,
            "half_life_s": half_life,
            "n_bubbles_rendered": len(centers),
            "mean_area_um2": mean_area,
            "std_area_um2": std_area,
            "bc_mm2": bc,
            "ravg_um": float(row.get("Ravg [µm]", np.nan)),
            "r32_um": float(row.get("R32 [µm]", np.nan)),
        })

    pd.DataFrame(rows).to_csv(out_dir / "frames_metadata.csv", index=False)
    return half_life

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bd", required=True, help="BD.xlsx path")
    ap.add_argument("--hd", required=True, help="HD.xlsx path")
    ap.add_argument("--out", default="data/synth/run1", help="output folder")
    ap.add_argument("--nframes", type=int, default=40)
    args = ap.parse_args()

    hl = generate_sequence(args.bd, args.hd, args.out, args.nframes)
    print("✅ Generated synthetic frames in:", args.out)
    print("Foam half-life (s):", hl)

