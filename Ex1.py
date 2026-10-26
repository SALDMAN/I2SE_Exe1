"""Ex1 demo: project a 3D NRRD volume onto a list of planes and write an MP4.

Usage:
    python Ex1.py <file.nrrd> <file.csv> <file.mp4> [--strict]
                  [--interp {nearest,trilinear}] [--window LO HI] [--fps N]

Pipeline:
    file.csv  -> load_planes -> valid Plane3D list
    file.nrrd -> pynrrd      -> NumPy volume M[x][y][z]
    slice_volume(plane, M) -> 2D image -> Frame -> Video -> file.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nrrd
import numpy as np

from Algorithem.Plane3D  import InvalidPlaneError
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
    if data.dtype.type not in SUPPORTED_DTYPES:
        raise ValueError(
            f"unsupported voxel type {data.dtype}; expected uint8, uint16 or uint32"
        )
    return data


# --interp name -> scipy.ndimage interpolation order used by slice_volume
INTERP_ORDER = {"nearest": 0, "trilinear": 1}


def window_bounds(volume: np.ndarray, window: tuple[float, float] | None):
    """Return (lo, hi) used to map the volume to 8 bits, or None for uint8 data."""
    if window is not None:
        lo, hi = window
        if hi <= lo:
            raise ValueError("--window requires LO < HI")
        return lo, hi
    if volume.dtype == np.uint8:
        return None  # already 8 bit, keep values untouched
    return float(volume.min()), float(volume.max())


def to_uint8(img: np.ndarray, bounds) -> np.ndarray:
    """Linearly window an image to uint8 (identity for uint8 data)."""
    if bounds is None:
        return img
    lo, hi = bounds
    scale = 255.0 / (hi - lo) if hi > lo else 0.0
    out = (img.astype(np.float32) - lo) * scale
    return np.clip(out, 0, 255).astype(np.uint8)


def pad_center(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Centre `img` on a black (rows x cols) canvas; all video frames must match."""
    h, w = img.shape
    if (h, w) == (rows, cols):
        return img
    canvas = np.zeros((rows, cols), dtype=img.dtype)
    r0, c0 = (rows - h) // 2, (cols - w) // 2
    canvas[r0:r0 + h, c0:c0 + w] = img
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
    bounds = window_bounds(volume, tuple(args.window) if args.window else None)
    order = INTERP_ORDER[args.interp]
    video = Video(fps=args.fps)

    t_proj = 0.0
    for plane in planes:
        t0 = time.perf_counter()
        image = slice_volume(plane, volume, order=order)
        t_proj += time.perf_counter() - t0
        frame_img = pad_center(to_uint8(image, bounds), rows, cols)
        video.add_frame(frame_img)

    avg_ms = 1000.0 * t_proj / len(planes)
    print(f"projection: {avg_ms:.2f} ms/frame ({1000.0 / avg_ms:.0f} FPS)")

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