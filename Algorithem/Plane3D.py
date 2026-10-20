"""A plane in 3D space defined by an explicit normal vector and a point on the plane."""

from __future__ import annotations

import math

Vec3 = tuple[float, float, float]

EPS = 1e-9


# ---------- small vector helpers ----------
def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


class Plane:
    """
    A plane defined by a specific normal vector and a point it passes through.

    The normal is stored exactly as you give it (it is NOT rescaled).
    Use `unit_normal` when you need a normalized version.
    Plane equation:  nx*x + ny*y + nz*z + d = 0,  where d = -(normal . point).
    """

    def __init__(self, normal: Vec3, point: Vec3 = (0.0, 0.0, 0.0)):
        normal = tuple(float(v) for v in normal)
        if len(normal) != 3:
            raise ValueError("normal must have exactly 3 components")
        if _length(normal) < EPS:
            raise ValueError("The normal vector cannot be the zero vector")
        self.normal: Vec3 = normal
        self.point: Vec3 = tuple(float(v) for v in point)
        self.d: float = -_dot(self.normal, self.point)

    # ---------- alternative constructors ----------
    @classmethod
    def from_points(cls, p1: Vec3, p2: Vec3, p3: Vec3) -> "Plane":
        """Plane through three points. Normal = (p2 - p1) x (p3 - p1)."""
        normal = _cross(_sub(p2, p1), _sub(p3, p1))
        if _length(normal) < EPS:
            raise ValueError("The three points are collinear; they do not define a plane")
        return cls(normal, p1)

    @classmethod
    def from_coefficients(cls, a: float, b: float, c: float, d: float) -> "Plane":
        """Plane from the equation a*x + b*y + c*z + d = 0 (normal = (a, b, c))."""
        n = (a, b, c)
        n2 = _dot(n, n)
        if n2 < EPS:
            raise ValueError("(a, b, c) cannot be zero")
        return cls(n, _mul(n, -d / n2))

    # ---------- normal ----------
    @property
    def unit_normal(self) -> Vec3:
        return _mul(self.normal, 1 / _length(self.normal))

    def flip(self) -> "Plane":
        """Return the same plane with the normal pointing the opposite way."""
        return Plane(_mul(self.normal, -1), self.point)

    # ---------- moving the plane ----------
    def move(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "Plane":
        """Move the plane by (dx, dy, dz). The normal stays the same. Returns self."""
        self.point = _add(self.point, (float(dx), float(dy), float(dz)))
        self.d = -_dot(self.normal, self.point)
        return self

    def move_along_normal(self, distance: float) -> "Plane":
        """Move the plane `distance` units along its normal (negative = opposite way). Returns self."""
        return self.move(*_mul(self.unit_normal, distance))

    def move_to(self, point: Vec3) -> "Plane":
        """Place the plane so it passes through `point`. The normal stays the same. Returns self."""
        self.point = tuple(float(v) for v in point)
        self.d = -_dot(self.normal, self.point)
        return self

    # ---------- rotating the plane ----------
    @staticmethod
    def _rotate_vector(v: Vec3, axis: Vec3, angle_deg: float) -> Vec3:
        """Rotate vector `v` around `axis` by `angle_deg` (Rodrigues' formula)."""
        k = _mul(axis, 1 / _length(axis))
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        return _add(
            _add(_mul(v, cos_a), _mul(_cross(k, v), sin_a)),
            _mul(k, _dot(k, v) * (1 - cos_a)),
        )

    def rotate(self, axis: Vec3, angle_deg: float, center: Vec3 | None = None) -> "Plane":
        """
        Rotate the plane by `angle_deg` degrees around `axis` (right-hand rule).

        The normal vector is rotated (keeping its length), and so is the plane's
        reference point around `center`. If `center` is None the plane pivots
        around its own point. Returns self.
        """
        if _length(axis) < EPS:
            raise ValueError("The rotation axis cannot be the zero vector")
        center = self.point if center is None else tuple(float(v) for v in center)
        self.normal = self._rotate_vector(self.normal, axis, angle_deg)
        self.point = _add(center, self._rotate_vector(_sub(self.point, center), axis, angle_deg))
        self.d = -_dot(self.normal, self.point)
        return self

    def rotate_around_normal(self, angle_deg: float, center: Vec3 | None = None) -> "Plane":
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
        return (*self.normal, self.d)

    def signed_distance(self, p: Vec3) -> float:
        """Positive on the side the normal points to, negative on the other side."""
        return (_dot(self.normal, p) + self.d) / _length(self.normal)

    def distance(self, p: Vec3) -> float:
        return abs(self.signed_distance(p))

    def contains(self, p: Vec3, tol: float = 1e-9) -> bool:
        return self.distance(p) <= tol

    def side(self, p: Vec3, tol: float = 1e-9) -> int:
        """+1 in front of the plane (normal side), -1 behind it, 0 on the plane."""
        s = self.signed_distance(p)
        return 0 if abs(s) <= tol else (1 if s > 0 else -1)

    # ---------- point operations ----------
    def project_point(self, p: Vec3) -> Vec3:
        """Closest point on the plane to `p`."""
        return _sub(p, _mul(self.unit_normal, self.signed_distance(p)))

    def reflect_point(self, p: Vec3) -> Vec3:
        """Mirror image of `p` across the plane."""
        return _sub(p, _mul(self.unit_normal, 2 * self.signed_distance(p)))

    # ---------- relations with other objects ----------
    def intersect_line(self, origin: Vec3, direction: Vec3) -> Vec3 | None:
        """Intersection with the line origin + t*direction, or None if parallel."""
        denom = _dot(self.normal, direction)
        if abs(denom) < EPS:
            return None
        t = -(_dot(self.normal, origin) + self.d) / denom
        return _add(origin, _mul(direction, t))

    def intersect_plane(self, other: "Plane") -> tuple[Vec3, Vec3] | None:
        """Intersection line with another plane as (point, unit direction), or None if parallel."""
        u = _cross(self.normal, other.normal)
        u2 = _dot(u, u)
        if u2 < EPS:
            return None
        h1, h2 = -self.d, -other.d  # n . p = h
        point = _mul(
            _add(_mul(_cross(other.normal, u), h1), _mul(_cross(u, self.normal), h2)),
            1 / u2,
        )
        return point, _mul(u, 1 / math.sqrt(u2))

    def angle_with(self, other: "Plane") -> float:
        """Angle between the planes in degrees, in [0, 90]."""
        cos_a = abs(_dot(self.unit_normal, other.unit_normal))
        return math.degrees(math.acos(min(1.0, cos_a)))

    def is_parallel(self, other: "Plane", tol: float = 1e-9) -> bool:
        return _length(_cross(self.unit_normal, other.unit_normal)) <= tol

    def is_perpendicular(self, other: "Plane", tol: float = 1e-9) -> bool:
        return abs(_dot(self.unit_normal, other.unit_normal)) <= tol

    def __repr__(self) -> str:
        return f"Plane(normal={self.normal}, point={self.point})"


