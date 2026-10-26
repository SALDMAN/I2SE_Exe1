"""Plane3D: a plane in 3D space, optionally bounded by an isosceles triangle / trapezoid.

* As an *infinite* plane it is defined by a normal vector and a point
  (``Plane3D(normal, point)``, ``from_points``, ``from_coefficients``).
* As a *region* for the projection algorithm it also remembers the four corners
  ``a1, a2, b, c`` (``from_triangle``, ``from_trapezoid``, ``from_corners``).
  Row ``y`` of the output image runs between a point walking from ``a1`` to
  ``b`` and a point walking from ``a2`` to ``c``. A triangle (a, b, c) is the
  special case ``a1 = a2 = a``.

Moving / rotating the plane moves the corners with it, so the region always
stays consistent with the plane.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vec3 = NDArray[np.float64]

EPS = 1e-9          # numeric safety (zero-vector checks, parallel lines, ...)
REGION_EPS = 1e-3   # tolerance of the assignment's Q&A (equal sides, coplanar)


class InvalidPlaneError(ValueError):
    """The given points do not define a valid plane region."""


def _vec3(v: ArrayLike, name: str = "vector") -> Vec3:
    """Convert to a float array of shape (3,), always as a copy."""
    a = np.array(v, dtype=float)
    if a.shape != (3,):
        raise ValueError(f"{name} must have exactly 3 components")
    return a


class Plane3D:
    """
    A plane defined by a specific normal vector and a point it passes through,
    optionally bounded by a region (see module docstring).

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
        self._corners: NDArray[np.float64] | None = None  # rows: a1, a2, b, c

    # ---------- alternative constructors (infinite plane) ----------
    @classmethod
    def from_points(cls, p1: ArrayLike, p2: ArrayLike, p3: ArrayLike) -> "Plane3D":
        """Plane through three points. Normal = (p2 - p1) x (p3 - p1)."""
        p1, p2, p3 = _vec3(p1), _vec3(p2), _vec3(p3)
        normal = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(normal) < EPS:
            raise ValueError("The three points are collinear; they do not define a plane")
        return cls(normal, p1)

    @classmethod
    def from_coefficients(cls, a: float, b: float, c: float, d: float) -> "Plane3D":
        """Plane from the equation a*x + b*y + c*z + d = 0 (normal = (a, b, c))."""
        n = np.array([a, b, c], dtype=float)
        n2 = float(n @ n)
        if n2 < EPS:
            raise ValueError("(a, b, c) cannot be zero")
        return cls(n, n * (-d / n2))

    # ---------- alternative constructors (bounded region) ----------
    @classmethod
    def from_triangle(cls, a: ArrayLike, b: ArrayLike, c: ArrayLike) -> "Plane3D":
        """Isosceles triangle: |dist(a,b) - dist(a,c)| < 0.001."""
        return cls.from_trapezoid(a, a, b, c)

    @classmethod
    def from_trapezoid(
        cls, a1: ArrayLike, a2: ArrayLike, b: ArrayLike, c: ArrayLike
    ) -> "Plane3D":
        """
        Isosceles trapezoid with the sides a1->b and a2->c:
          |dist(a1,b) - dist(a2,c)| < 0.001,  dist(a1,a2) <= dist(b,c),
          a1, a2, b, c on the same plane.
        """
        try:
            corners = np.array(
                [_vec3(a1, "a1"), _vec3(a2, "a2"), _vec3(b, "b"), _vec3(c, "c")]
            )
        except ValueError as e:
            raise InvalidPlaneError(str(e)) from e
        if not np.all(np.isfinite(corners)):
            raise InvalidPlaneError("coordinates must be finite")
        a1, a2, b, c = corners
        d_left = float(np.linalg.norm(b - a1))
        d_right = float(np.linalg.norm(c - a2))
        d_top = float(np.linalg.norm(a2 - a1))
        d_bc = float(np.linalg.norm(c - b))

        if abs(d_left - d_right) >= REGION_EPS:
            raise InvalidPlaneError(
                f"the two sides must have equal length: {d_left:.4f} vs {d_right:.4f}")
        if d_left < REGION_EPS or d_bc < REGION_EPS:
            raise InvalidPlaneError("degenerate plane (zero length side)")
        if d_top > d_bc + REGION_EPS:
            raise InvalidPlaneError(
                f"dist(a1,a2)={d_top:.4f} must be <= dist(b,c)={d_bc:.4f}")

        normal = np.cross(b - a1, c - a1)
        n_len = float(np.linalg.norm(normal))
        if n_len < REGION_EPS:
            raise InvalidPlaneError("degenerate plane (collinear points)")
        off_plane = abs(float((a2 - a1) @ normal)) / n_len
        if off_plane >= REGION_EPS:
            raise InvalidPlaneError(f"points are not coplanar (off by {off_plane:.4f})")

        plane = cls(normal, corners.mean(axis=0))
        plane._corners = corners
        return plane

    @classmethod
    def from_corners(cls, points: Sequence[ArrayLike]) -> "Plane3D":
        """3 points (a, b, c) or 4 points (a1, a2, b, c), e.g. one CSV row."""
        if len(points) == 3:
            return cls.from_triangle(*points)
        if len(points) == 4:
            return cls.from_trapezoid(*points)
        raise InvalidPlaneError(f"a plane needs 3 or 4 points, got {len(points)}")

    # ---------- region (needs a bounded plane) ----------
    @property
    def has_region(self) -> bool:
        return self._corners is not None

    def _need_region(self) -> NDArray[np.float64]:
        if self._corners is None:
            raise ValueError("this plane has no region (build it with from_triangle / from_trapezoid)")
        return self._corners

    @property
    def a1(self) -> Vec3:
        return self._need_region()[0].copy()

    @property
    def a2(self) -> Vec3:
        return self._need_region()[1].copy()

    @property
    def b(self) -> Vec3:
        return self._need_region()[2].copy()

    @property
    def c(self) -> Vec3:
        return self._need_region()[3].copy()

    @property
    def length(self) -> float:
        """Dab: distance a1->b (number of rows of the image)."""
        a1, _, b, _ = self._need_region()
        return float(np.linalg.norm(b - a1))

    @property
    def width(self) -> float:
        """Dbc: distance b->c (width of the widest row)."""
        _, _, b, c = self._need_region()
        return float(np.linalg.norm(c - b))

    @property
    def v_left(self) -> Vec3:
        """Unit step of the left row end (a1 -> b)."""
        a1, _, b, _ = self._need_region()
        return (b - a1) / np.linalg.norm(b - a1)

    @property
    def v_right(self) -> Vec3:
        """Unit step of the right row end (a2 -> c)."""
        _, a2, _, c = self._need_region()
        return (c - a2) / np.linalg.norm(c - a2)

    @property
    def rows(self) -> int:
        """Image height: round(Dab), at least 1."""
        return max(1, round(self.length))

    @property
    def cols(self) -> int:
        """Image width: round(Dbc), at least 1."""
        return max(1, round(self.width))

    def copy(self) -> "Plane3D":
        """Independent copy (the move/rotate methods change the object in place)."""
        p = Plane3D(self.normal, self.point)
        p._corners = None if self._corners is None else self._corners.copy()
        return p

    # ---------- normal ----------
    @property
    def unit_normal(self) -> Vec3:
        return self.normal / np.linalg.norm(self.normal)

    def flip(self) -> "Plane3D":
        """Return the same plane with the normal pointing the opposite way."""
        p = Plane3D(-self.normal, self.point)
        p._corners = None if self._corners is None else self._corners.copy()
        return p

    # ---------- moving the plane ----------
    def _update_d(self) -> None:
        self.d = -float(self.normal @ self.point)

    def _shift(self, delta: Vec3) -> None:
        self.point = self.point + delta
        if self._corners is not None:
            self._corners = self._corners + delta
        self._update_d()

    def move(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Plane3D":
        """Move the plane (and its region) by (dx, dy, dz). The normal stays the same. Returns self."""
        self._shift(np.array([dx, dy, dz], dtype=float))
        return self

    def move_along_normal(self, distance: float) -> "Plane3D":
        """Move the plane `distance` units along its normal (negative = opposite way). Returns self."""
        self._shift(self.unit_normal * distance)
        return self

    def move_to(self, point: ArrayLike) -> "Plane3D":
        """Place the plane so it passes through `point`. The normal stays the same. Returns self."""
        self._shift(_vec3(point, "point") - self.point)
        return self

    # ---------- rotating the plane ----------
    @staticmethod
    def _rotate_vector(v: ArrayLike, axis: ArrayLike, angle_deg: float) -> Vec3:
        """Rotate vector(s) `v` (shape (..., 3)) around `axis` by `angle_deg` (Rodrigues' formula)."""
        v = np.asarray(v, dtype=float)
        axis = np.asarray(axis, dtype=float)
        k = axis / np.linalg.norm(axis)
        a = np.radians(angle_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)
        dots = np.asarray(v @ k)[..., None]
        return v * cos_a + np.cross(k, v) * sin_a + k * dots * (1 - cos_a)

    def rotate(
        self, axis: ArrayLike, angle_deg: float, center: ArrayLike | None = None
    ) -> "Plane3D":
        """
        Rotate the plane by `angle_deg` degrees around `axis` (right-hand rule).

        The normal vector is rotated (keeping its length), and so are the plane's
        reference point and its region corners, around `center`. If `center` is
        None the plane pivots around its own point. Returns self.
        """
        axis = _vec3(axis, "axis")
        if np.linalg.norm(axis) < EPS:
            raise ValueError("The rotation axis cannot be the zero vector")
        center = self.point.copy() if center is None else _vec3(center, "center")
        self.normal = self._rotate_vector(self.normal, axis, angle_deg)
        self.point = center + self._rotate_vector(self.point - center, axis, angle_deg)
        if self._corners is not None:
            self._corners = center + self._rotate_vector(self._corners - center, axis, angle_deg)
        self._update_d()
        return self

    def rotate_around_normal(self, angle_deg: float, center: ArrayLike | None = None) -> "Plane3D":
        """
        Rotate the plane around its own normal vector.

        An infinite plane looks the same after spinning around its normal, but a
        plane with a region does change: its corners spin, so the projected image
        rotates in the plane. With the default center nothing else moves.
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

    def intersect_plane(self, other: "Plane3D") -> tuple[Vec3, Vec3] | None:
        """Intersection line with another plane as (point, unit direction), or None if parallel."""
        u = np.cross(self.normal, other.normal)
        u2 = float(u @ u)
        if u2 < EPS:
            return None
        h1, h2 = -self.d, -other.d  # n . p = h
        point = (np.cross(other.normal, u) * h1 + np.cross(u, self.normal) * h2) / u2
        return point, u / np.sqrt(u2)

    def angle_with(self, other: "Plane3D") -> float:
        """Angle between the planes in degrees, in [0, 90]."""
        cos_a = abs(float(self.unit_normal @ other.unit_normal))
        return float(np.degrees(np.arccos(min(1.0, cos_a))))

    def is_parallel(self, other: "Plane3D", tol: float = 1e-9) -> bool:
        return bool(np.linalg.norm(np.cross(self.unit_normal, other.unit_normal)) <= tol)

    def is_perpendicular(self, other: "Plane3D", tol: float = 1e-9) -> bool:
        return bool(abs(self.unit_normal @ other.unit_normal) <= tol)

    def __repr__(self) -> str:
        if self._corners is None:
            return f"Plane3D(normal={self.normal.tolist()}, point={self.point.tolist()})"
        a1, a2, b, c = (x.tolist() for x in self._corners)
        return f"Plane3D(a1={a1}, a2={a2}, b={b}, c={c})"