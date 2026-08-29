"""Geometry primitives shared by the calibration schemas (REQ-4, REQ-7).

Points and polygons come in two flavors: `PixelPoint` (raw image pixel
space, used only for homography source correspondences) and `TablePoint`
(the table plane's own coordinate system, used everywhere geometry is
actually decided — REQ-5). Nothing below `detection` may accept pixel
coordinates; that boundary is enforced by which type a field declares.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, Field

from poker_vision.schema_base import StrictModel


class PixelPoint(StrictModel):
    """A point in raw image pixel space (origin top-left, x right, y down)."""

    x: float
    y: float


class TablePoint(StrictModel):
    """A point in table-plane coordinates, in the table's own unit (see `TableDimensions.unit`)."""

    x: float
    y: float


class TablePolygon(StrictModel):
    """A closed polygon in table coordinates.

    Points are listed in order; the closing edge (last point back to the
    first) is implicit and must not be repeated.
    """

    points: list[TablePoint] = Field(min_length=3)


class TableUnit(StrEnum):
    MM = "mm"
    CM = "cm"
    M = "m"


class TableDimensions(StrictModel):
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    unit: TableUnit


class ImageDimensions(StrictModel):
    """Pixel dimensions of the image a calibration was authored/compiled against."""

    width: int = Field(gt=0)
    height: int = Field(gt=0)


def _check_3x3(value: list[list[float]]) -> list[list[float]]:
    if len(value) != 3 or any(len(row) != 3 for row in value):
        raise ValueError("expected a 3x3 matrix (3 rows of 3 floats)")
    return value


Matrix3x3 = Annotated[list[list[float]], AfterValidator(_check_3x3)]
