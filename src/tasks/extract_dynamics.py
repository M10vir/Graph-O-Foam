from __future__ import annotations

from pathlib import Path
import re
import argparse

import pandas as pd
import cv2
import numpy as np

from src.core.bubbles import detect_bubbles, bubble_stats, draw_overlay


def parse_time_from_name(name: str):
    m = re.search(r"_t([0-9]+(?:\.[0-9]+)?)", name)
    return float(m.group(1)) if m else None


def normalize_bubbles(b):
    """
    Normalize output of detect_bubbles() to what bubble_stats/draw_overlay expect:
    list[dict] with at least keys: x, y, r, and ideally 'circ' or 'circularity'.
    """
    if b is None:
        return []

    # detect_bubbles may return (bubbles, debug)
    if isinstance(b, tuple) and len(b) >= 1:
        b = b[0]

    if not b:
        return []

    # If already dict bubbles, just ensure circ key exists
    if isinstance(b[0], dict):
        out = []
        for d in b:
            dd = dict(d)
            # your detector uses 'circularity' key
            if "circ" not in dd and "circularity" in dd:
                dd["circ"] = dd["circularity"]
            out.append(dd)
        return out

    # Otherwise convert tuple/list/ndarray bubbles like [x,y,r,(circ)]
    out = []
    for item in b:
        if isinstance(item, dict):
            dd = dict(item)
            if "circ" not in dd and "circularity" in dd:
                dd["circ"] = dd["circularity"]
            out.append(dd)
            continue

        if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 3:
            dd = {"x": float(item[0]), "y": float(item[1]), "r": float(item[2])}
            dd["circ"] = float(item[3]) if len(item) >= 4 else float("nan")
            out.append(dd)

    return out


def run(folder: str, out_csv: str | None = None, out_overlays: str | None = None):
    folder_p = Path(folder)
    frames = sorted(folder_p.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {folder_p} (expected frame_*.png)")

    out_csv_path = Path(out_csv) if out_csv else (folder_p / "bubble_dynamics.csv")

    overlays_dir = Path(out_overlays) if out_overlays else (folder_p / "overlays")
    overlays_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bubbles_raw = detect_bubbles(gray)
        bubbles = normalize_bubbles(bubbles_raw)

        stats = bubble_stats(bubbles)
        t_s = parse_time_from_name(f.name)

        row = {"frame": f.name, "t_s": t_s}
        if isinstance(stats, dict):
            row.update(stats)
        rows.append(row)

        overlay = draw_overlay(img, bubbles)
        cv2.imwrite(str(overlays_dir / f.name), overlay)

    df = pd.DataFrame(rows)
    if "t_s" in df.columns and df["t_s"].notna().any():
        df = df.sort_values(["t_s", "frame"])
    else:
        df = df.sort_values(["frame"])

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)

    return str(out_csv_path), str(overlays_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Run folder containing frame_*.png")
    ap.add_argument("--out-csv", default=None, help="Default: <folder>/bubble_dynamics.csv")
    ap.add_argument("--overlays", default=None, help="Default: <folder>/overlays")
    args = ap.parse_args()

    overlays_path = args.overlays if args.overlays else str(Path(args.folder) / "overlays")

    out_csv, out_ov = run(args.folder, out_csv=args.out_csv, out_overlays=overlays_path)
    print("✅ Saved dynamics:", out_csv)
    print("✅ Saved overlays in:", out_ov)
 
