"""
Convert foam image + tabular measurements to NSID-compatible .h5 files for DTMicroscope.

This script looks for folders next to this repo that contain:
- PNG frames (any depth of nested subfolders)
- A "*BD.xlsx" file (bubble descriptors)
- A "*HD.xlsx" file (height / volume traces)

Each output .h5 contains:
- Channel_000: image stack (time, y, x)
- Channel_001: tabular features aligned to the frames (time, feature_index)

Original metadata carries uuid links so SciFiReaders + DTMicroscope can load them.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import pyNSID
import sidpy
from PIL import Image
from scipy import ndimage as ndi


# -----------------------------------------------------------------------------
# Utility helpers


def otsu_threshold(values: np.ndarray) -> float:
    """Simple Otsu threshold on a float image in [0, 1]."""
    hist, bin_edges = np.histogram(values.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = values.size
    cumulative_sum = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256))
    global_mean = cumulative_mean[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = (global_mean * cumulative_sum - cumulative_mean) ** 2
        denominator = cumulative_sum * (total - cumulative_sum)
        sigma_b_squared = numerator / denominator

    sigma_b_squared = np.nan_to_num(sigma_b_squared)
    max_index = np.argmax(sigma_b_squared)
    return bin_edges[max_index]


def estimate_lamella_thickness_mm(
    gray_image: np.ndarray, pixel_size_mm: float, max_size: int = 512
) -> float:
    """
    Estimate lamella thickness from a grayscale image.

    Steps:
    - downsample to at most max_size on the longer edge to keep computation light
    - invert (lamellae are dark)
    - Otsu threshold to get lamella mask
    - distance transform -> thickness ~ 2 * 95th percentile of distance map
    """
    img = gray_image
    if img.ndim == 3:
        img = img.mean(axis=2)
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0

    # downsample keeping aspect
    h, w = img.shape
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        new_shape = (int(h * scale), int(w * scale))
        img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(new_shape[::-1], Image.BILINEAR)) / 255.0

    inverted = 1.0 - img
    thresh = otsu_threshold(inverted)
    mask = inverted > thresh
    if not mask.any():
        return float("nan")

    dist = ndi.distance_transform_edt(mask)
    thickness_px = 2.0 * np.percentile(dist[mask], 95)
    return float(thickness_px * pixel_size_mm)


def parse_composition(folder_name: str) -> List[Dict[str, str]]:
    """
    Extract surfactant + wt% pairs from a folder name.
    Example: '0.5 wt% coco-betain & 0.5wt % Capryl-glucoside'
    """
    cleaned = folder_name.replace("_", " ").replace("%", "% ").replace("  ", " ")
    parts = re.split(r"&", cleaned)
    comps: List[Dict[str, str]] = []
    pattern = re.compile(r"([\d.]+)\s*wt\s*%?\s*([A-Za-z0-9 \-]+)", re.IGNORECASE)
    for part in parts:
        match = pattern.search(part.strip())
        if match:
            comps.append(
                {
                    "concentration_wt_percent": float(match.group(1)),
                    "surfactant": match.group(2).strip(),
                }
            )
    return comps


def compute_half_life(time_s: np.ndarray, signal: np.ndarray) -> float:
    """
    Half-life defined as first time the signal drops below half of its max.
    Returns NaN if not reached.
    """
    if len(signal) == 0:
        return float("nan")
    start = np.nanmax(signal)
    if start <= 0:
        return float("nan")
    target = 0.5 * start
    below = np.where(signal <= target)[0]
    return float(time_s[below[0]]) if len(below) > 0 else float("nan")


def interpolate_hd(hd_df: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
    """Interpolate HD measurements to the BD/frame time base."""
    if hd_df is None or hd_df.empty:
        return pd.DataFrame(index=range(len(times)))
    hd_df = hd_df.dropna()
    if "t [s]" not in hd_df.columns or hd_df.empty:
        return pd.DataFrame(index=range(len(times)))
    base_time = hd_df["t [s]"].to_numpy()
    if base_time.size == 0:
        return pd.DataFrame(index=range(len(times)))
    interp_data = {}
    for col in hd_df.columns:
        if col == "t [s]":
            continue
        values = hd_df[col].to_numpy()
        interp_data[col] = np.interp(times, base_time, values)
    return pd.DataFrame(interp_data)


def load_frame_paths(root: Path) -> List[Path]:
    """Return sorted list of png frames under a folder."""
    return sorted(root.rglob("*.png"))


def load_image_stack(frame_paths: Sequence[Path]) -> np.ndarray:
    """Load images into a uint8 stack (time, y, x)."""
    frames: List[np.ndarray] = []
    for path in frame_paths:
        arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        frames.append(arr)
    return np.stack(frames, axis=0)


def save_dataset_dictionary(h5_file: h5py.File, datasets: Dict[str, object]) -> None:
    """Persist sidpy datasets + dicts to NSID format (adapted from DT notebooks)."""
    h5_measurement_group = sidpy.hdf.prov_utils.create_indexed_group(h5_file, "Measurement_")
    for key, dataset in datasets.items():
        if key.endswith("/"):
            key = key[:-1]
        if isinstance(dataset, sidpy.Dataset):
            h5_group = h5_measurement_group.create_group(key)
            h5_dataset = pyNSID.hdf_io.write_nsid_dataset(dataset, h5_group)
            dataset.h5_dataset = h5_dataset
            h5_dataset.file.flush()
        elif isinstance(dataset, dict):
            sidpy.hdf.hdf_utils.write_dict_to_h5_group(h5_measurement_group, dataset, key)
        else:
            print(f"Could not save item {key} of dataset dictionary")


@dataclass
class FoamDataset:
    name: str
    folder: Path
    bd_file: Path
    hd_file: Optional[Path]
    frame_paths: List[Path]


def find_datasets(raw_root: Path) -> List[FoamDataset]:
    """Identify foam datasets under raw_root."""
    datasets: List[FoamDataset] = []
    for folder in raw_root.iterdir():
        if not folder.is_dir() or folder.name.startswith("DTMicroscope-main"):
            continue
        bd_files = list(folder.glob("*BD.xlsx"))
        if not bd_files:
            continue
        frame_paths = load_frame_paths(folder)
        if not frame_paths:
            continue
        hd_files = list(folder.glob("*HD.xlsx"))
        datasets.append(
            FoamDataset(
                name=folder.name,
                folder=folder,
                bd_file=bd_files[0],
                hd_file=hd_files[0] if hd_files else None,
                frame_paths=frame_paths,
            )
        )
    return datasets


def build_nsid_file(
    foam: FoamDataset,
    output_path: Path,
    pixel_size_mm: float,
    max_frames: Optional[int] = None,
) -> None:
    """Convert one dataset into an NSID .h5 file."""
    bd_df = pd.read_excel(foam.bd_file)
    times = bd_df["t [s]"].to_numpy()
    frame_paths = foam.frame_paths
    frame_count = min(len(frame_paths), len(times))
    frame_paths = frame_paths[:frame_count]
    times = times[:frame_count]
    bd_df = bd_df.iloc[:frame_count]
    if max_frames:
        frame_paths = frame_paths[:max_frames]
        times = times[: len(frame_paths)]
        bd_df = bd_df.iloc[: len(frame_paths)]

    hd_df = pd.read_excel(foam.hd_file) if foam.hd_file else None
    hd_interp = interpolate_hd(hd_df, times) if hd_df is not None else pd.DataFrame(index=range(len(times)))

    # Image stack + lamella thickness
    image_stack = load_image_stack(frame_paths)
    lamella_mm = []
    for frame in image_stack:
        lamella_mm.append(estimate_lamella_thickness_mm(frame, pixel_size_mm))
    lamella_mm = np.array(lamella_mm)

    feature_df = bd_df.reset_index(drop=True)
    if not hd_interp.empty:
        feature_df = pd.concat([feature_df, hd_interp.reset_index(drop=True)], axis=1)
    feature_df["lamella_thickness_mm"] = lamella_mm

    half_life_col = None
    if hd_interp is not None and "hfoam [mm]" in hd_interp.columns:
        half_life_col = "hfoam [mm]"
    elif "BC [mm⁻²]" in feature_df.columns:
        half_life_col = "BC [mm⁻²]"
    half_life = compute_half_life(times, feature_df[half_life_col].to_numpy()) if half_life_col else float("nan")
    feature_df["half_life_s"] = half_life

    composition = parse_composition(foam.name)
    metadata = {
        "dataset_name": foam.name,
        "source_folder": str(foam.folder),
        "bd_file": str(foam.bd_file),
        "hd_file": str(foam.hd_file) if foam.hd_file else None,
        "composition": composition,
        "pixel_size_mm": pixel_size_mm,
        "half_life_s": half_life,
        "lamella_method": "otsu+distance_transform (95th percentile)",
    }

    # Build sidpy datasets
    image_uuid = str(uuid.uuid4())
    feature_uuid = str(uuid.uuid4())

    im_ds = sidpy.Dataset.from_array(image_stack, name="foam_frames")
    im_ds.data_type = sidpy.DataType.IMAGE
    im_ds.units = "a.u."
    im_ds.quantity = "intensity"
    im_ds.original_metadata.update(metadata)
    im_ds.original_metadata["uuid"] = image_uuid
    im_ds.set_dimension(
        0,
        sidpy.Dimension(times, name="time", units="s", quantity="Time", dimension_type="temporal"),
    )
    im_ds.set_dimension(
        1,
        sidpy.Dimension(
            np.arange(image_stack.shape[1]) * pixel_size_mm,
            name="y",
            units="mm",
            quantity="Length",
            dimension_type="spatial",
        ),
    )
    im_ds.set_dimension(
        2,
        sidpy.Dimension(
            np.arange(image_stack.shape[2]) * pixel_size_mm,
            name="x",
            units="mm",
            quantity="Length",
            dimension_type="spatial",
        ),
    )

    feature_ds = sidpy.Dataset.from_array(feature_df.to_numpy(dtype=np.float32), name="foam_features")
    feature_ds.data_type = sidpy.DataType.SPECTRUM
    feature_ds.units = "mixed"
    feature_ds.original_metadata.update(metadata)
    feature_ds.original_metadata["uuid"] = feature_uuid
    feature_ds.original_metadata["associated-image"] = image_uuid
    feature_ds.metadata["feature_names"] = feature_df.columns.tolist()
    feature_ds.set_dimension(
        0,
        sidpy.Dimension(times, name="time", units="s", quantity="Time", dimension_type="temporal"),
    )
    feature_ds.set_dimension(
        1,
        sidpy.Dimension(
            np.arange(feature_df.shape[1]),
            name="feature_index",
            units="index",
            quantity="Feature",
            dimension_type="spectral",
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_f:
        h5_f.attrs["description"] = "Foam dataset converted for DTMicroscope"
        h5_f.attrs["composition"] = json.dumps(composition)
        save_dataset_dictionary(
            h5_f,
            {
                "Channel_000": im_ds,
                "Channel_001": feature_ds,
                "metadata": metadata,
                "columns": {"names": feature_df.columns.tolist()},
            },
        )
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert foam data to NSID .h5 files")
    parser.add_argument("--raw-root", type=Path, default=Path("."), help="Folder that contains the raw foam folders")
    parser.add_argument("--output-dir", type=Path, default=Path("DTMicroscope-main/data/foam"), help="Destination folder for .h5 files")
    parser.add_argument("--pixel-size-mm", type=float, default=0.01, help="Physical size per pixel (mm)")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap for quick runs")
    args = parser.parse_args()

    datasets = find_datasets(args.raw_root)
    if not datasets:
        print("No datasets found. Expect folders with PNG frames and '*BD.xlsx'.")
        return

    for ds in datasets:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", ds.name).strip("_").lower()
        output_path = args.output_dir / f"{slug}.h5"
        build_nsid_file(ds, output_path, pixel_size_mm=args.pixel_size_mm, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
