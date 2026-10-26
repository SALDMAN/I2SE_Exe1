import numpy as np
from scipy.ndimage import map_coordinates
from Plane3D import Plane3D

def slice_volume(plane, volume, spacing=None, origin=(0.0, 0.0, 0.0), order=1, cval=0.0):
    rows, cols = plane.rows, plane.cols
    a1, a2, b, c = plane.a1, plane.a2, plane.b, plane.c

    t = (np.arange(rows) + 0.5) / rows
    s = (np.arange(cols) + 0.5) / cols
    left = a1 + t[:, None] * (b - a1)
    right = a2 + t[:, None] * (c - a2)
    pts = left[:, None, :] + s[None, :, None] * (right - left)[:, None, :]

    if spacing is not None:
        pts = (pts - np.asarray(origin)) / np.asarray(spacing)

    coords = np.moveaxis(pts, -1, 0)
    return map_coordinates(volume, coords, order=order, mode="constant", cval=cval)