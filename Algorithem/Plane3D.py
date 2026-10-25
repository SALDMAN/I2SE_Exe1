
"""A plane in 3D space defined by an explicit normal vector and a point on the plane (NumPy version)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vec3 = NDArray[np.float64]

EPS = 1e-9


def _vec3(v: ArrayLike, name: str = "vector") -> Vec3:
    """Convert to a float array of shape (3,), always as a copy."""
    a = np.array(v, dtype=float)
    if a.shape != (3,):
        raise ValueError(f"{name} must have exactly 3 components")
    return a


class Plane:
    """
    A plane defined by a specific normal vector and a point it passes through.

    The normal is stored exactly as you give it (it is NOT rescaled).
    Use `unit_normal` when you need a normalized version.
    Plane equation:  nx*x + ny*y + nz*z + d = 0,  where d = -(normal . point).

    Point-based queries (`signed_distance`, `project_point`, `reflect_point`, ...)
    accept either a single point of shape (3,) or an array of points of shape (..., 3).
    """

    def __init__(self, normal: ArrayLike, point: ArrayLike = (0.0, 0.0, 0.0)):
        normal = _vec3(normal, "normal")
        if np.linalg.norm(normal) < EPS:
            raise ValueError("The normal vector cannot be the zero vector")
        self.normal: Vec3 = normal
        self.point: Vec3 = _vec3(point, "point")
        self.d: float = -float(self.normal @ self.point)

    # ---------- alternative constructors ----------
    @classmethod
    def from_points(cls, p1: ArrayLike, p2: ArrayLike, p3: ArrayLike) -> "Plane":
        """Plane through three points. Normal = (p2 - p1) x (p3 - p1)."""
        p1, p2, p3 = _vec3(p1), _vec3(p2), _vec3(p3)
        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) < EPS:
            raise ValueError("The three points are collinear; they do not define a plane")
        return cls(normal, p1)

    @classmethod
    def from_coefficients(cls, a: float, b: float, c: float, d: float) -> "Plane":
        """Plane from the equation a*x + b*y + c*z + d = 0 (normal = (a, b, c))."""
        n = np.array([a, b, c], dtype=float)
        n2 = float(n @ n)
        if n2 < EPS:
            raise ValueError("(a, b, c) cannot be zero")
        return cls(n, n * (-d / n2))

    # ---------- normal ----------
    @property
    def unit_normal(self) -> Vec3:
        return self.normal / np.linalg.norm(self.normal)

    def flip(self) -> "Plane":
        """Return the same plane with the normal pointing the opposite way."""
        return Plane(-self.normal, self.point)

    # ---------- moving the plane ----------
    def _update_d(self) -> None:
        self.d = -float(self.normal @ self.point)

    def move(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Plane":
        """Move the plane by (dx, dy, dz). The normal stays the same. Returns self."""
        self.point = self.point + np.array([dx, dy, dz], dtype=float)
        self._update_d()
        return self

    def move_along_normal(self, distance: float) -> "Plane":
        """Move the plane `distance` units along its normal (negative = opposite way). Returns self."""
        self.point = self.point + self.unit_normal * distance
        self._update_d()
        return self

    def move_to(self, point: ArrayLike) -> "Plane":
        """Place the plane so it passes through `point`. The normal stays the same. Returns self."""
        self.point = _vec3(point, "point")
        self._update_d()
        return self

    # ---------- rotating the plane ----------
    @staticmethod
    def _rotate_vector(v: ArrayLike, axis: ArrayLike, angle_deg: float) -> Vec3:
        """Rotate vector `v` around `axis` by `angle_deg` (Rodrigues' formula)."""
        v = np.asarray(v, dtype=float)
        axis = np.asarray(axis, dtype=float)
        k = axis / np.linalg.norm(axis)
        a = np.radians(angle_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)
        return v * cos_a + np.cross(k, v) * sin_a + k * (k @ v) * (1 - cos_a)

    def rotate(
        self, axis: ArrayLike, angle_deg: float, center: ArrayLike | None = None
    ) -> "Plane":
        """
        Rotate the plane by `angle_deg` degrees around `axis` (right-hand rule).

        The normal vector is rotated (keeping its length), and so is the plane's
        reference point around `center`. If `center` is None the plane pivots
        around its own point. Returns self.
        """
        axis = _vec3(axis, "axis")
        if np.linalg.norm(axis) < EPS:
            raise ValueError("The rotation axis cannot be the zero vector")
        center = self.point.copy() if center is None else _vec3(center, "center")
        self.normal = self._rotate_vector(self.normal, axis, angle_deg)
        self.point = center + self._rotate_vector(self.point - center, axis, angle_deg)
        self._update_d()
        return self

    def rotate_around_normal(self, angle_deg: float, center: ArrayLike | None = None) -> "Plane":
        """
        Rotate the plane around its own normal vector.

        An infinite plane looks the same after spinning around its normal, so with
        the default center nothing changes. With a `center` that is off the plane's
        axis, the reference point orbits that center (the plane itself stays the same).
        """
        return self.rotate(self.normal, angle_deg, center)

    # ---------- queries ----------
    @property
    def coefficients(self) -> tuple[float, float, float, float]:
        return (*map(float, self.normal), self.d)

    def signed_distance(self, p: ArrayLike) -> float | NDArray[np.float64]:
        """Positive on the side the normal points to, negative on the other side."""
        p = np.asarray(p, dtype=float)
        return (p @ self.normal + self.d) / np.linalg.norm(self.normal)

    def distance(self, p: ArrayLike) -> float | NDArray[np.float64]:
        return np.abs(self.signed_distance(p))

    def contains(self, p: ArrayLike, tol: float = 1e-9) -> bool | NDArray[np.bool_]:
        return self.distance(p) <= tol

    def side(self, p: ArrayLike, tol: float = 1e-9) -> int | NDArray[np.int_]:
        """+1 in front of the plane (normal side), -1 behind it, 0 on the plane."""
        s = np.asarray(self.signed_distance(p))
        out = np.where(np.abs(s) <= tol, 0, np.sign(s)).astype(int)
        return int(out) if out.ndim == 0 else out

    # ---------- point operations ----------
    def project_point(self, p: ArrayLike) -> Vec3:
        """Closest point on the plane to `p` (or to each row of `p`)."""
        p = np.asarray(p, dtype=float)
        s = np.asarray(self.signed_distance(p))
        return p - s[..., None] * self.unit_normal

    def reflect_point(self, p: ArrayLike) -> Vec3:
        """Mirror image of `p` (or of each row of `p`) across the plane."""
        p = np.asarray(p, dtype=float)
        s = np.asarray(self.signed_distance(p))
        return p - 2 * s[..., None] * self.unit_normal

    # ---------- relations with other objects ----------
    def intersect_line(self, origin: ArrayLike, direction: ArrayLike) -> Vec3 | None:
        """Intersection with the line origin + t*direction, or None if parallel."""
        origin, direction = _vec3(origin, "origin"), _vec3(direction, "direction")
        denom = float(self.normal @ direction)
        if abs(denom) < EPS:
            return None
        t = -(float(self.normal @ origin) + self.d) / denom
        return origin + direction * t

    def intersect_plane(self, other: "Plane") -> tuple[Vec3, Vec3] | None:
        """Intersection line with another plane as (point, unit direction), or None if parallel."""
        u = np.cross(self.normal, other.normal)
        u2 = float(u @ u)
        if u2 < EPS:
            return None
        h1, h2 = -self.d, -other.d  # n . p = h
        point = (np.cross(other.normal, u) * h1 + np.cross(u, self.normal) * h2) / u2
        return point, u / np.sqrt(u2)

    def angle_with(self, other: "Plane") -> float:
        """Angle between the planes in degrees, in [0, 90]."""
        cos_a = abs(float(self.unit_normal @ other.unit_normal))
        return float(np.degrees(np.arccos(min(1.0, cos_a))))

    def is_parallel(self, other: "Plane", tol: float = 1e-9) -> bool:
        return bool(np.linalg.norm(np.cross(self.unit_normal, other.unit_normal)) <= tol)

    def is_perpendicular(self, other: "Plane", tol: float = 1e-9) -> bool:
        return bool(abs(self.unit_normal @ other.unit_normal) <= tol)

    def __repr__(self) -> str:
        return f"Plane(normal={self.normal.tolist()}, point={self.point.tolist()})"
