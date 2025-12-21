# src/tasks/extract_graph_metrics.py
from __future__ import annotations

import argparse
from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd

from src.graph.graph_builder import delaunay_edges, compute_graph_stats


def _parse_time_from_name(filename: str) -> float | None:
    # matches ..._t150.png or ..._t150.0.png
    m = re.search(r"_t([0-9]+(?:\.[0-9]+)?)", filename)
    return float(m.group(1)) if m else None


def _load_t_map_from_dynamics(run_dir: Path) -> dict[str, float]:
    """
    Optional: align graph metrics to Phase-1 bubble_dynamics.csv timeline.
    Expected columns: frame, t_s
    """
    dyn_csv = run_dir / "bubble_dynamics.csv"
    if not dyn_csv.exists():
        return {}

    try:
        df = pd.read_csv(dyn_csv)
        if "frame" in df.columns and "t_s" in df.columns:
            out = {}
            for _, r in df[["frame", "t_s"]].dropna().iterrows():
                out[str(r["frame"])] = float(r["t_s"])
            return out
    except Exception:
        pass

    return {}


def detect_bubbles_centroids(gray: np.ndarray) -> np.ndarray:
    """
    Centroid detection for synthetic microscopy frames (Phase-1 outputs).
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If foreground is tiny, invert
    if (th > 0).mean() < 0.25:
        th = cv2.bitwise_not(th)

    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pts = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 30:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        pts.append([cx, cy])

    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def draw_graph_overlay(bgr: np.ndarray, points: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    out = bgr.copy()

    for u, v in edges:
        p1 = tuple(np.round(points[u]).astype(int))
        p2 = tuple(np.round(points[v]).astype(int))
        cv2.line(out, p1, p2, (0, 255, 255), 1)

    for p in points:
        p = tuple(np.round(p).astype(int))
        cv2.circle(out, p, 2, (0, 0, 255), -1)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Run folder containing frame_*.png (Phase-1 output)")
    ap.add_argument("--out", default=None, help="Output CSV path (default: <folder>/graph_metrics.csv)")
    ap.add_argument("--overlay-every", type=int, default=5, help="Save overlay every N frames (default: 5)")
    ap.add_argument("--no-overlays", action="store_true", help="Disable saving graph overlays")
    args = ap.parse_args()

    run_dir = Path(args.folder)
    if not run_dir.exists():
        raise FileNotFoundError(f"Folder not found: {run_dir}")

    frames = sorted(run_dir.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {run_dir} (expected frame_*.png)")

    out_csv = Path(args.out) if args.out else run_dir / "graph_metrics.csv"

    t_map = _load_t_map_from_dynamics(run_dir)  # uses Phase-1 CSV if available

    overlay_dir = run_dir / "graph_overlays"
    if not args.no_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, fp in enumerate(frames):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        pts = detect_bubbles_centroids(gray)
        edges = delaunay_edges(pts) if len(pts) >= 2 else []
        stats = compute_graph_stats(pts, edges)

        # Timeline alignment: prefer Phase-1 bubble_dynamics.csv, else filename _t###
        t_s = t_map.get(fp.name, _parse_time_from_name(fp.name))

        rows.append(
            {
                "frame": fp.name,
                "t_s": t_s,
                "n_nodes": stats.n_nodes,
                "n_edges": stats.n_edges,
                "avg_degree": stats.avg_degree,
                "density": stats.density,
                "avg_edge_len": stats.avg_edge_len,
                "std_edge_len": stats.std_edge_len,
                "n_components": stats.n_components,
                "giant_component_ratio": stats.giant_component_ratio,
            }
        )

        # Save overlays (throttled)
        if (not args.no_overlays) and (idx % max(1, args.overlay_every) == 0):
            ov = draw_graph_overlay(bgr, pts, edges)
            cv2.imwrite(str(overlay_dir / fp.name), ov)

    df = pd.DataFrame(rows)
    if "t_s" in df.columns and df["t_s"].notna().any():
        df = df.sort_values(["t_s", "frame"])
    else:
        df = df.sort_values(["frame"])

    df.to_csv(out_csv, index=False)

    print(f"✅ Saved graph metrics: {out_csv}")
    if not args.no_overlays:
        print(f"✅ Saved graph overlays in: {overlay_dir}")


if __name__ == "__main__":
    main()
 
