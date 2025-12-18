from pathlib import Path
import re
import pandas as pd
import cv2

from src.core.bubbles import detect_bubbles, bubble_stats, draw_overlay

def parse_time_from_name(name: str):
    m = re.search(r"_t(\d+)", name)
    return float(m.group(1)) if m else None

def run(folder="data/synth/coco_1wt_all", out_csv=None, out_overlays=None):
    folder = Path(folder)
    frames = sorted(folder.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {folder}")

    if out_csv is None:
        out_csv = folder / "bubble_dynamics.csv"
    else:
        out_csv = Path(out_csv)

    if out_overlays:
        out_overlays = Path(out_overlays)
        out_overlays.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in frames:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        bubbles, _mask = detect_bubbles(img)
        stats = bubble_stats(bubbles)
        t = parse_time_from_name(f.name)

        if out_overlays:
            overlay = draw_overlay(img, bubbles)
            cv2.imwrite(str(out_overlays / f.name), overlay)

        rows.append({
            "frame": f.name,
            "t_s": t,
            **stats
        })

    df = pd.DataFrame(rows).sort_values("t_s")
    df.to_csv(out_csv, index=False)
    return out_csv

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data/synth/coco_1wt_all")
    ap.add_argument("--overlays", default="data/synth/coco_1wt_all/overlays")
    args = ap.parse_args()

    out = run(args.folder, out_overlays=args.overlays)
    print("✅ Saved dynamics:", out)
    print("✅ Saved overlays in:", args.overlays)
