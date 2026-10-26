"""Tests that a Plane3D is created (or rejected) correctly."""
import numpy as np
import pytest

from Algorithem.io_csv import PlaneCsvReader
from Algorithem.Plane3D import InvalidPlaneError, Plane3D
from Algorithem.Plane_loader import build_planes

# ----------------------------------------------------------- sample points
# (taken from the assignment's sample sheet)
TRI = [(200, 200, 0), (0, 200, 400), (400, 200, 400)]                   # plane y = 200
RECT = [(200, 0, 0), (200, 0, 500), (200, 400, 0), (200, 400, 500)]    # file order a1,b,a2,c


# ------------------------------------------------------- plain (infinite) plane
def test_plane_from_normal_and_point():
    p = Plane3D([0, 0, 2], [1, 1, 5])
    assert not p.has_region
    assert p.d == pytest.approx(-10)                       # -(normal . point)
    assert np.allclose(p.unit_normal, [0, 0, 1])
    assert p.signed_distance([0, 0, 8]) == pytest.approx(3)


def test_plane_zero_normal_rejected():
    with pytest.raises(ValueError):
        Plane3D([0, 0, 0])


def test_plane_from_three_points():
    p = Plane3D.from_points([0, 0, 0], [1, 0, 0], [0, 1, 0])
    assert np.allclose(p.unit_normal, [0, 0, 1])


def test_plane_from_collinear_points_rejected():
    with pytest.raises(ValueError):
        Plane3D.from_points([0, 0, 0], [1, 1, 1], [2, 2, 2])


# ------------------------------------------------------------ triangle region
def test_triangle_is_created():
    p = Plane3D.from_triangle(*TRI)
    assert p.has_region
    assert np.allclose(p.a1, p.a2)                 # triangle: a1 == a2
    assert np.allclose(p.a1, TRI[0])
    assert np.allclose(p.b, TRI[1])
    assert np.allclose(p.c, TRI[2])
    assert p.length == pytest.approx(np.hypot(200, 400))   # Dab
    assert p.width == pytest.approx(400)                   # Dbc
    assert (p.rows, p.cols) == (447, 400)
    assert np.allclose(np.abs(p.unit_normal), [0, 1, 0])   # plane y = 200


def test_triangle_step_vectors_are_unit_length():
    p = Plane3D.from_triangle(*TRI)
    assert np.linalg.norm(p.v_left) == pytest.approx(1)
    assert np.linalg.norm(p.v_right) == pytest.approx(1)


def test_triangle_unequal_sides_rejected():
    with pytest.raises(InvalidPlaneError, match="equal length"):
        Plane3D.from_triangle((0, 0, 0), (10, 0, 0), (0, 12, 0))


def test_triangle_equal_sides_within_eps_accepted():
    Plane3D.from_triangle((0, 0, 0), (10, 0, 0), (0, 10.0005, 0))


def test_triangle_equal_sides_outside_eps_rejected():
    with pytest.raises(InvalidPlaneError):
        Plane3D.from_triangle((0, 0, 0), (10, 0, 0), (0, 10.002, 0))


def test_degenerate_triangle_rejected():
    with pytest.raises(InvalidPlaneError):
        Plane3D.from_triangle((1, 1, 1), (1, 1, 1), (1, 1, 1))       # zero length
    with pytest.raises(InvalidPlaneError):
        Plane3D.from_triangle((0, 0, 0), (1, 0, 0), (-1, 0, 0))      # collinear


# ----------------------------------------------------------- trapezoid region
def test_trapezoid_is_created():
    p = Plane3D.from_trapezoid(*[RECT[0], RECT[2], RECT[1], RECT[3]])  # a1,a2,b,c
    assert p.has_region
    assert not np.allclose(p.a1, p.a2)
    assert (p.rows, p.cols) == (500, 400)
    assert np.allclose(p.v_left, [0, 0, 1])
    assert np.allclose(p.v_right, [0, 0, 1])


def test_trapezoid_unequal_sides_rejected():
    with pytest.raises(InvalidPlaneError, match="equal length"):
        Plane3D.from_trapezoid((0, 0, 0), (4, 0, 0), (0, 0, 10), (4, 0, 12))


def test_trapezoid_top_wider_than_bottom_rejected():
    with pytest.raises(InvalidPlaneError, match="must be <="):
        Plane3D.from_trapezoid((0, 0, 0), (10, 0, 0), (2, 0, 10), (8, 0, 10))


def test_trapezoid_not_coplanar_rejected():
    # equal sides (10 and 10) and a1,a2 close enough, but a2 is 3 units off the plane a1,b,c
    c = (4 + np.sqrt(55), 0, 6)
    with pytest.raises(InvalidPlaneError, match="coplanar"):
        Plane3D.from_trapezoid((0, 0, 0), (4, 3, 0), (0, 0, 10), c)


# ----------------------------------------------------------- from_corners
def test_from_corners_accepts_3_and_4_points():
    assert Plane3D.from_corners(TRI).has_region
    assert Plane3D.from_corners([RECT[0], RECT[2], RECT[1], RECT[3]]).has_region


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_from_corners_wrong_number_of_points(n):
    with pytest.raises(InvalidPlaneError, match="3 or 4"):
        Plane3D.from_corners([(0, 0, 0)] * n)


def test_non_finite_or_malformed_points_rejected():
    with pytest.raises(InvalidPlaneError):
        Plane3D.from_triangle((0, 0, 0), (np.nan, 0, 1), (1, 0, 1))
    with pytest.raises(InvalidPlaneError):
        Plane3D.from_triangle((0, 0), (1, 0, 1), (1, 0, 1))


# ------------------------------------------------------ region needs a region
def test_region_properties_need_a_region():
    p = Plane3D([0, 0, 1])
    with pytest.raises(ValueError, match="no region"):
        _ = p.rows


# ---------------------------------------------------- the whole CSV -> Plane3D
SAMPLE_CSV = """Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4
3,200,200,0,0,200,400,400,200,400,,,
4,200,0,0,200,0,500,200,400,0,200,400,500
3,200,200,-30,-50,200,400,400,250,400,,,
4,200,50,0,200,0,500,200,350,0,200,400,500
"""


def test_sample_csv_creates_planes():
    result = build_planes(PlaneCsvReader().parse(SAMPLE_CSV.splitlines()))
    sizes = [(p.rows, p.cols) for p in result.planes]
    assert sizes == [(447, 400), (500, 400), (502, 400)]
    # the third data row (file line 4) has sides of different length
    assert [e.line for e in result.errors] == [4]
    assert "equal length" in result.errors[0].message


# ------------------------------------------------------------ copy / moving
def test_copy_is_independent_and_moving_keeps_region_valid():
    p = Plane3D.from_triangle(*TRI)
    q = p.copy().rotate([0, 0, 1], 37).move(5, -3, 2)
    assert np.allclose(p.a1, TRI[0])                      # original untouched
    assert q.length == pytest.approx(p.length)
    assert q.width == pytest.approx(p.width)
    assert np.allclose(q.distance(np.array([q.a1, q.b, q.c])), 0, atol=1e-9)
    Plane3D.from_corners([q.a1, q.b, q.c])                # still a valid region