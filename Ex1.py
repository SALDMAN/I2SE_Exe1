"""Ex1 demo: project a 3D NRRD volume onto a list of planes and write an MP4.

Usage:
    python Ex1.py <file.nrrd> <file.csv> <file.mp4> [--strict]
                  [--interp {nearest,trilinear}] [--window LO HI] [--fps N]
                  [--workers N]

Pipeline:
    file.csv  -> load_planes -> valid Plane3D list
    file.nrrd -> pynrrd      -> NumPy volume M[x][y][z]
    slice_volume(plane, M) -> 2D image -> Frame -> Video -> file.mp4
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import time

import nrrd
import numpy as np
from Algorithem.Plane3D import InvalidPlaneError
from Algorithem.Plane_loader import load_planes
from Algorithem.sampler import slice_volume
from Algorithem.video import Frame, Video

SUPPORTED_DTYPES = (np.uint8, np.uint16, np.uint32)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Project a 3D NRRD volume onto planes from a CSV and render an MP4."
    )
    p.add_argument("nrrd", type=Path, nargs="?",
                   help="input 3D volume (.nrrd); default: first .nrrd found")
    p.add_argument("csv", type=Path, nargs="?",
                   help="input planes (.csv); default: first .csv found")
    p.add_argument("mp4", type=Path, nargs="?", default=Path("results/out.mp4"),
                   help="output video (.mp4); default: results/out.mp4")
    p.add_argument("--strict", action="store_true",
                   help="abort on the first invalid plane (default: skip and warn)")
    p.add_argument("--interp", choices=("nearest", "trilinear"), default="nearest",
                   help="sampling method (default: nearest)")
    p.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"),
                   help="intensity window for 16/32-bit data (default: global min/max)")
    p.add_argument("--fps", type=int, default=30, help="video frame rate (default: 30)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                   help="number of parallel projection threads (default: CPU count)")
    return p.parse_args(argv)


def find_first(pattern: str) -> Path | None:
    """First file matching `pattern` in the script folder, then in its data/ folder."""
    here = Path(__file__).resolve().parent
    for folder in (here, here / "data"):
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_volume(path: Path) -> np.ndarray:
    """Load a 3D unsigned-integer NRRD as M[x][y][z] (voxel space, no copy later)."""
    if not path.is_file():
        raise FileNotFoundError(f"NRRD file not found: {path}")
    data, _header = nrrd.read(str(path))
    if data.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {data.ndim}D array")
    if data.dtype.kind == "i":
        # Signed integers (e.g. int16 MRHead): shift so the minimum becomes 0,
        # then store as the unsigned type of the same size. Relative intensities
        # are preserved, which is all the projection and the 8-bit window need.
        unsigned = np.dtype(f"uint{data.dtype.itemsize * 8}")
        data = (data.astype(np.int64) - int(data.min())).astype(unsigned)
        print(f"note: signed volume shifted to {unsigned}")
    if data.dtype.type not in SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported voxel type {data.dtype}; expected an integer type"
        )
    return data


# --interp name -> scipy.ndimage interpolation order used by slice_volume
INTERP_ORDER = {"nearest": 0, "trilinear": 1}


def get_window_params(volume: np.ndarray, window: tuple[float, float] | None) -> tuple[float, float] | None:
    """Return precomputed (lo, scale) for fast 8-bit mapping, or None for uint8."""
    if window is not None:
        lo, hi = window
        if hi <= lo:
            raise ValueError("--window requires LO < HI")
        scale = 255.0 / (hi - lo)
        return float(lo), scale

    if volume.dtype == np.uint8:
        return None

    lo = float(volume.min())
    hi = float(volume.max())
    scale = 255.0 / (hi - lo) if hi > lo else 0.0
    return lo, scale


def process_single_plane(plane, volume: np.ndarray, order: int,
                         win_params: tuple[float, float] | None,
                         rows: int, cols: int) -> np.ndarray:
    """Worker task: slices, windows, and centres plane onto the canvas."""
    img = slice_volume(plane, volume, order=order)

    # Window to uint8
    if win_params is not None:
        lo, scale = win_params
        # Fast in-place operations on float32 representation
        out_f = (img.astype(np.float32) - lo) * scale
        img_u8 = np.clip(out_f, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img_u8 = img.astype(np.uint8)
    else:
        img_u8 = img

    # Fast canvas placement without intermediate copies
    h, w = img_u8.shape
    if (h, w) == (rows, cols):
        return img_u8

    canvas = np.zeros((rows, cols), dtype=np.uint8)
    r0 = (rows - h) // 2
    c0 = (cols - w) // 2
    canvas[r0:r0 + h, c0:c0 + w] = img_u8
    return canvas


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    # 0. No file names given: look for them next to this script / in data/.
    args.nrrd = args.nrrd or find_first("*.nrrd")
    args.csv = args.csv or find_first("*.csv")
    if args.nrrd is None or args.csv is None:
        print("error: could not find a .nrrd and a .csv file. Put them next to "
              "Ex1.py (or in a data/ folder), or pass their paths.", file=sys.stderr)
        return 1
    print(f"using volume: {args.nrrd}")
    print(f"using planes: {args.csv}")

    # 1. Planes (validated). Strict mode raises on the first problem.
    try:
        result = load_planes(args.csv, strict=args.strict)
        if isinstance(result, tuple):          # (planes, errors)
            planes, errors = result
        else:                                  # PlaneLoadResult object
            planes = list(result.planes)
            errors = list(getattr(result, "errors", []))
    except (InvalidPlaneError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for err in errors:
        print(f"warning: skipped invalid plane: {err}", file=sys.stderr)
    if not planes:
        print("error: no valid planes in CSV, nothing to render", file=sys.stderr)
        return 2

    # 2. Volume.
    try:
        volume = load_volume(args.nrrd)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"volume: shape={volume.shape} dtype={volume.dtype}")
    print(f"planes: {len(planes)} valid, {len(errors)} skipped")

    # 3. Projection. All frames must share one size, so use the largest plane.
    rows = max(p.rows for p in planes)
    cols = max(p.cols for p in planes)
    win_params = get_window_params(volume, tuple(args.window) if args.window else None)
    order = INTERP_ORDER[args.interp]
    video = Video(fps=args.fps)

    print(f"projecting frames using {args.workers} worker threads...")
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        frames = list(pool.map(
            lambda p: process_single_plane(p, volume, order, win_params, rows, cols),
            planes
        ))

    total_proj_time = time.perf_counter() - t_start
    avg_ms = (total_proj_time * 1000.0) / len(planes)
    effective_fps = len(planes) / total_proj_time if total_proj_time > 0 else 0.0
    print(f"projection: {avg_ms:.2f} ms/frame ({effective_fps:.1f} FPS)")

    for frame in frames:
        video.add_frame(frame)

    # 4. Export.
    args.mp4.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    video.to_mp4(str(args.mp4), args.fps)
    print(f"wrote {len(planes)} frames ({cols}x{rows}) to {args.mp4} "
          f"in {time.perf_counter() - t0:.1f} s")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())