"""Reading plane definitions from a CSV file.

Expected format (see the assignment's sample sheet)::

    Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4
    3,200,200,0,0,200,400,400,200,400,,,
    4,200,0,0,200,0,500,200,400,0,200,400,500

* The first row may be a header (skipped when its first cell is not a number).
* The first column is the number of points (3 or 4); then 3 numbers per point.
* Empty trailing cells and empty rows are ignored.

This module only turns text into numbers. It does NOT check the geometry
(isosceles, coplanar, ...). That is the job of ``geometry.Plane``.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

Point = tuple[float, float, float]

# Canonical order used by the rest of the program.
THREE_POINT_ORDER = ("a", "b", "c")
FOUR_POINT_ORDER = ("a1", "a2", "b", "c")

# Order of the 4 points as they appear in the file. The sample file seems to use
# a1, b, a2, c (a1->b and a2->c are the two equal vectors). Change here if the
# course staff say otherwise.
DEFAULT_FOUR_POINT_FILE_ORDER = ("a1", "b", "a2", "c")


@dataclass(frozen=True)
class PlaneRow:
    """One CSV row: raw points, in canonical order (a,b,c) or (a1,a2,b,c)."""
    line: int
    points: tuple[Point, ...]


@dataclass(frozen=True)
class RowError:
    line: int
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: {self.message}"


@dataclass
class CsvResult:
    rows: list[PlaneRow] = field(default_factory=list)
    errors: list[RowError] = field(default_factory=list)


class CsvFormatError(Exception):
    """Raised for unreadable files, or for bad rows in strict mode."""


class PlaneCsvReader:
    def __init__(self, four_point_file_order: Sequence[str] = DEFAULT_FOUR_POINT_FILE_ORDER):
        if sorted(four_point_file_order) != sorted(FOUR_POINT_ORDER):
            raise ValueError(f"four_point_file_order must be a permutation of {FOUR_POINT_ORDER}")
        # index in the file of each canonical point name
        self._four_idx = [four_point_file_order.index(n) for n in FOUR_POINT_ORDER]

    # ---------------------------------------------------------------- public
    def read(self, path: str, strict: bool = False) -> CsvResult:
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                return self.parse(f, strict=strict)
        except OSError as e:
            raise CsvFormatError(f"cannot read '{path}': {e}") from e

    def parse(self, lines: Iterable[str], strict: bool = False) -> CsvResult:
        """Parse CSV text (any iterable of lines). Bad rows go to ``errors``,
        or raise CsvFormatError at the first one when ``strict`` is True."""
        result = CsvResult()
        first_data_row = True
        reader = csv.reader(lines)
        for cells in reader:
            line = reader.line_num
            cells = self._strip_trailing_empty(cells)
            if not cells:
                continue  # empty row
            if first_data_row:
                first_data_row = False
                if not self._is_number(cells[0]):
                    continue  # header
            try:
                result.rows.append(PlaneRow(line, self._parse_row(cells)))
            except ValueError as e:
                err = RowError(line, str(e))
                if strict:
                    raise CsvFormatError(str(err)) from e
                result.errors.append(err)
        return result

    # --------------------------------------------------------------- private
    def _parse_row(self, cells: list[str]) -> tuple[Point, ...]:
        count = self._parse_count(cells[0])
        values = cells[1:]
        if len(values) != 3 * count:
            raise ValueError(
                f"expected {3 * count} coordinates for {count} points, got {len(values)}")
        nums = [self._to_float(v) for v in values]
        pts = [tuple(nums[i:i + 3]) for i in range(0, len(nums), 3)]
        if count == 4:
            pts = [pts[i] for i in self._four_idx]
        return tuple(pts)  # type: ignore[arg-type]

    @staticmethod
    def _parse_count(text: str) -> int:
        try:
            count = int(float(text))
            if count != float(text):
                raise ValueError
        except ValueError:
            raise ValueError(f"invalid number of points '{text}'") from None
        if count not in (3, 4):
            raise ValueError(f"number of points must be 3 or 4, got {count}")
        return count

    @staticmethod
    def _to_float(text: str) -> float:
        text = text.strip()
        if text == "":
            raise ValueError("missing coordinate value")
        try:
            x = float(text)
        except ValueError:
            raise ValueError(f"not a number: '{text}'") from None
        if not math.isfinite(x):
            raise ValueError(f"coordinate must be finite: '{text}'")
        return x

    @staticmethod
    def _is_number(text: str) -> bool:
        try:
            float(text)
            return True
        except ValueError:
            return False

    @staticmethod
    def _strip_trailing_empty(cells: list[str]) -> list[str]:
        cells = [c.strip() for c in cells]
        while cells and cells[-1] == "":
            cells.pop()
        return cells