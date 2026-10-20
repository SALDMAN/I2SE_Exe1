"""A mathematical plane in 3D space: a*x + b*y + c*z + d = 0 (no external dependencies)."""

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
    An infinite plane defined by the equation  a*x + b*y + c*z + d = 0.

    The normal (a, b, c) is always stored as a unit vector, so
    `signed_distance` returns true geometric distances.
    """

    def __init__(self, a: float, b: float, c: float, d: float):
        length = _length((a, b, c))
        if length < EPS:
            raise ValueError("The normal vector (a, b, c) cannot be zero")
        self.normal: Vec3 = (a / length, b / length, c / length)
        self.d: float = d / length

    # ---------- alternative constructors ----------
    @classmethod
    def from_point_normal(cls, point: Vec3, normal: Vec3) -> "Plane":
        """Plane passing through `point` and perpendicular to `normal`."""
        a, b, c = normal
        return cls(a, b, c, -_dot(normal, point))

    @classmethod
    def from_points(cls, p1: Vec3, p2: Vec3, p3: Vec3) -> "Plane":
        """Plane passing through three non-collinear points."""
        normal = _cross(_sub(p2, p1), _sub(p3, p1))
        if _length(normal) < EPS:
            raise ValueError("The three points are collinear; they do not define a plane")
        return cls.from_point_normal(p1, normal)

    # ---------- queries ----------
    @property
    def coefficients(self) -> tuple[float, float, float, float]:
        return (*self.normal, self.d)

    def signed_distance(self, point: Vec3) -> float:
        """Positive on the side the normal points to, negative on the other side."""
        return _dot(self.normal, point) + self.d

    def distance(self, point: Vec3) -> float:
        return abs(self.signed_distance(point))

    def contains(self, point: Vec3, tol: float = 1e-9) -> bool:
        return self.distance(point) <= tol

    def side(self, point: Vec3, tol: float = 1e-9) -> int:
        """+1 in front of the plane, -1 behind it, 0 on the plane."""
        s = self.signed_distance(point)
        return 0 if abs(s) <= tol else (1 if s > 0 else -1)

    def point_on_plane(self) -> Vec3:
        """Some point on the plane (the one closest to the origin)."""
        return _mul(self.normal, -self.d)

    # ---------- point operations ----------
    def project_point(self, point: Vec3) -> Vec3:
        """Closest point on the plane to `point`."""
        return _sub(point, _mul(self.normal, self.signed_distance(point)))

    def reflect_point(self, point: Vec3) -> Vec3:
        """Mirror image of `point` across the plane."""
        return _sub(point, _mul(self.normal, 2 * self.signed_distance(point)))

    # ---------- relations with other objects ----------
    def intersect_line(self, origin: Vec3, direction: Vec3) -> Vec3 | None:
        """
        Intersection of the plane with the line origin + t * direction.
        Returns None if the line is parallel to the plane (no single intersection point).
        """
        denom = _dot(self.normal, direction)
        if abs(denom) < EPS:
            return None
        t = -self.signed_distance(origin) / denom
        return _add(origin, _mul(direction, t))

    def intersect_plane(self, other: "Plane") -> tuple[Vec3, Vec3] | None:
        """
        Intersection line with another plane, as (point, direction).
        Returns None if the planes are parallel (or identical).
        """
        direction = _cross(self.normal, other.normal)
        denom = _dot(direction, direction)
        if denom < EPS:
            return None
        h1, h2 = -self.d, -other.d  # n . p = h
        point = _mul(
            _add(
                _mul(_cross(other.normal, direction), h1),
                _mul(_cross(direction, self.normal), h2),
            ),
            1 / denom,
        )
        return point, _mul(direction, 1 / math.sqrt(denom))

    def angle_with(self, other: "Plane") -> float:
        """Angle between the planes in degrees, in [0, 90]."""
        cos_a = abs(_dot(self.normal, other.normal))
        return math.degrees(math.acos(min(1.0, cos_a)))

    def is_parallel(self, other: "Plane", tol: float = 1e-9) -> bool:
        return _length(_cross(self.normal, other.normal)) <= tol

    def is_perpendicular(self, other: "Plane", tol: float = 1e-9) -> bool:
        return abs(_dot(self.normal, other.normal)) <= tol

    # ---------- dunder ----------
    def __repr__(self) -> str:
        a, b, c = self.normal
        return f"Plane({a:.4g}x + {b:.4g}y + {c:.4g}z + {self.d:.4g} = 0)"

