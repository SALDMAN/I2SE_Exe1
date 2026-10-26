"""Glue between the CSV reader and ``Plane3D``: file -> list of valid planes."""
from __future__ import annotations

from dataclasses import dataclass, field

from .Plane3D import InvalidPlaneError, Plane3D
from .io_csv import CsvFormatError, PlaneCsvReader, RowError


@dataclass
class PlaneLoadResult:
    planes: list[Plane3D] = field(default_factory=list)   # valid planes, in file order
    errors: list[RowError] = field(default_factory=list)  # bad rows (format or geometry)


def load_planes(path: str, strict: bool = False,
                reader: PlaneCsvReader | None = None) -> PlaneLoadResult:
    """Read ``path`` and build a Plane3D for every row.

    Invalid rows are collected in ``errors`` (and skipped), or raise
    ``CsvFormatError`` at the first one when ``strict`` is True.
    """
    reader = reader or PlaneCsvReader()
    return build_planes(reader.read(path, strict=strict), strict=strict)


def build_planes(csv_result, strict: bool = False) -> PlaneLoadResult:
    result = PlaneLoadResult(errors=list(csv_result.errors))
    for row in csv_result.rows:
        try:
            result.planes.append(Plane3D.from_corners(row.points))
        except InvalidPlaneError as e:
            err = RowError(row.line, str(e))
            if strict:
                raise CsvFormatError(str(err)) from e
            result.errors.append(err)
    result.errors.sort(key=lambda e: e.line)
    return result