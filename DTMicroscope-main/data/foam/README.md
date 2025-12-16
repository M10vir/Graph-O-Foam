# Foam datasets for DTMicroscope

Converted from the raw PNG stacks and Excel tables in `../..` using `scripts/convert_foam_to_h5.py`.

Contents:
- `*.h5`: NSID files with `Channel_000` (image stack, shape time × y × x) and `Channel_001` (tabular features aligned to frames). Original metadata holds composition, pixel size, half-life estimate, and lamella thickness method.
- `metadata/` (auto in H5): `metadata` group and `columns/names` list the feature order.

Key assumptions:
- Pixel size defaulted to `0.01 mm/pixel` (set `--pixel-size-mm` when re-running the converter if you have calibration).
- Half-life computed as first time height (`hfoam [mm]` if available, otherwise bubble count `BC`) falls below half of its maximum.
- Lamella thickness estimated per frame via inverted image → Otsu threshold → distance transform (95th percentile) scaled by pixel size.

Regenerate:
```bash
py scripts/convert_foam_to_h5.py --raw-root .. --output-dir DTMicroscope-main/data/foam --pixel-size-mm 0.01
```
