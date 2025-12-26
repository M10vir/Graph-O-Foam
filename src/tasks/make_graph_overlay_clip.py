from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def ffmpeg_path() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    # common Homebrew path on Apple Silicon
    fallback = "/opt/homebrew/bin/ffmpeg"
    if Path(fallback).exists():
        return fallback
    raise SystemExit("ffmpeg not found. Install with: brew install ffmpeg")


def run_cmd(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            "ffmpeg failed.\n"
            f"Command: {' '.join(cmd)}\n\n"
            f"STDERR:\n{e.stderr}\n"
        )


def escape_for_ffmpeg_glob(path: str) -> str:
    """
    ffmpeg -pattern_type glob treats [] and {} specially.
    Escape them so folders like 'suspension[69]' work.
    """
    return (
        path.replace("\\", "\\\\")
            .replace("[", r"\[")
            .replace("]", r"\]")
            .replace("{", r"\{")
            .replace("}", r"\}")
    )


def make_mp4_glob(overlay_dir: Path, out_mp4: Path, fps: int = 10, width: int = 1280) -> None:
    ff = ffmpeg_path()

    # IMPORTANT: escape glob-special chars in directory path
    pattern = escape_for_ffmpeg_glob(str((overlay_dir.resolve() / "*.png")))

    cmd = [
        ff, "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", pattern,
        "-vf", f"scale={width}:-2:flags=lanczos,format=yuv420p",
        "-c:v", "libx264",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    run_cmd(cmd)


def make_gif_glob(overlay_dir: Path, out_gif: Path, fps: int = 10, width: int = 900) -> None:
    ff = ffmpeg_path()

    pattern = escape_for_ffmpeg_glob(str((overlay_dir.resolve() / "*.png")))
    palette = out_gif.with_suffix(".palette.png")

    # palette
    cmd_palette = [
        ff, "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", pattern,
        "-vf", f"fps={fps},scale={width}:-2:flags=lanczos:force_original_aspect_ratio=decrease,palettegen",
        str(palette),
    ]
    run_cmd(cmd_palette)

    # gif
    cmd_gif = [
        ff, "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", pattern,
        "-i", str(palette),
        "-lavfi",
        f"fps={fps},scale={width}:-2:flags=lanczos:force_original_aspect_ratio=decrease[x];[x][1:v]paletteuse",
        str(out_gif),
    ]
    run_cmd(cmd_gif)

    if palette.exists():
        palette.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Run folder, e.g. data/synth/test_upload")
    ap.add_argument("--overlay-dir", default="graph_overlays", help="Subfolder name (default: graph_overlays)")
    ap.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    ap.add_argument("--width", type=int, default=1280, help="MP4 output width (default: 1280)")
    ap.add_argument("--gif", action="store_true", help="Also generate GIF")
    args = ap.parse_args()

    run_dir = Path(args.folder)
    overlay_dir = run_dir / args.overlay_dir
    if not overlay_dir.exists():
        raise SystemExit(f"Overlay folder not found: {overlay_dir}")

    pngs = sorted(overlay_dir.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No PNGs found in: {overlay_dir}")

    out_mp4 = run_dir / "graph_overlays.mp4"
    make_mp4_glob(overlay_dir, out_mp4, fps=max(1, int(args.fps)), width=int(args.width))
    print(f"✅ MP4 created: {out_mp4}")

    if args.gif:
        out_gif = run_dir / "graph_overlays.gif"
        make_gif_glob(overlay_dir, out_gif, fps=max(1, int(args.fps)), width=min(int(args.width), 900))
        print(f"✅ GIF created: {out_gif}")


if __name__ == "__main__":
    main() 
