import numpy as np
from scipy.ndimage import map_coordinates
from Plane3D import Plane3D

def slice_volume(
    plane: "Plane3D",
    volume: np.ndarray,
    spacing: tuple[float, float, float] | None = None,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """
    Cross-section of a 3D volume by a bounded plane.

    Returns a 2D array of shape (plane.rows, plane.cols).

    plane   : Plane3D with a region (from_triangle / from_trapezoid / from_corners).
    volume  : 3D array indexed [x, y, z].
    spacing : voxel size per axis. If None, plane coordinates are voxel indices.
              Otherwise plane coordinates are in physical units
              (e.g. mm) and are converted:  index = (coord - origin) / spacing.
    order   : interpolation order (0 = nearest, 1 = linear, 3 = cubic).
    cval    : value for points outside the volume.
    """
    if volume.ndim != 3:
        raise ValueError("volume must be 3D")
    if not plane.has_region:
        raise ValueError("plane must have a region (use from_triangle / from_trapezoid)")

    rows, cols = plane.rows, plane.cols
    a1, a2, b, c = plane.a1, plane.a2, plane.b, plane.c

    # pixel centers as fractions in [0, 1]
    t = (np.arange(rows) + 0.5) / rows          # position along the sides (a->b, a->c)
    s = (np.arange(cols) + 0.5) / cols          # position across a row

    left = a1 + t[:, None] * (b - a1)           # (rows, 3)
    right = a2 + t[:, None] * (c - a2)          # (rows, 3)

    # points[y, x] = left[y] + s[x] * (right[y] - left[y])
    pts = left[:, None, :] + s[None, :, None] * (right - left)[:, None, :]   # (rows, cols, 3)

    # physical coordinates -> voxel indices
    if spacing is not None:
        pts = (pts - np.asarray(origin)) / np.asarray(spacing)

    coords = np.moveaxis(pts, -1, 0)            # (3, rows, cols) as map_coordinates expects
    return map_coordinates(volume, coords, order=order, mode="constant", cval=cval)
