"""Pixel-space box math and the pixel -> table-plane transform (REQ-17).

`box_center` is the exact bounding-box-centre method verified in Phase 0
(`phase0_poc.py::Detection.center`): `((x1+x2)/2, (y1+y2)/2)`, nothing more.
Every detector implementation that derives a centre from a pixel box must
route through it so all detectors agree on the same method.

`apply_homography_to_point`/`transform_box_to_table` are the pixel -> table
transform itself, applied here in the detection stage so nothing behind it
ever sees pixel coordinates (REQ-5).
"""

from __future__ import annotations

from poker_vision.calibration.geometry import Matrix3x3, PixelPoint, TablePoint
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.detection.models import TableBoundingBox

PixelBox = tuple[float, float, float, float]
"""A pixel-space bounding box as (x1, y1, x2, y2), matching Phase 0's convention."""

# A homogeneous w this close to zero maps to (or past) the horizon -- not a
# point a top-down table camera can ever actually observe -- so treat it as
# an invalid transform rather than emit +/-inf table coordinates.
_W_EPSILON = 1e-9


def box_center(box: PixelBox) -> PixelPoint:
    """Exact bounding-box centre in pixel space (Phase 0's REQ-0.4 method)."""
    x1, y1, x2, y2 = box
    return PixelPoint(x=(x1 + x2) / 2.0, y=(y1 + y2) / 2.0)


def apply_homography_to_point(point: PixelPoint, homography: HomographyMatrix) -> TablePoint:
    """Map one pixel-space point onto the table plane via the forward homography."""
    x, y = _apply_matrix(homography.forward, point.x, point.y)
    return TablePoint(x=x, y=y)


def transform_box_to_table(box: PixelBox, homography: HomographyMatrix) -> TableBoundingBox:
    """Map a pixel-space box onto the table plane as an axis-aligned box.

    A homography can rotate/skew, so the four transformed corners are not
    generally axis-aligned in table space; the result is their bounding
    box, not a literal corner-to-corner mapping.
    """
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    transformed = [_apply_matrix(homography.forward, x, y) for x, y in corners]
    xs = [x for x, _ in transformed]
    ys = [y for _, y in transformed]
    return TableBoundingBox(
        min=TablePoint(x=min(xs), y=min(ys)),
        max=TablePoint(x=max(xs), y=max(ys)),
    )


def _apply_matrix(matrix: Matrix3x3, x: float, y: float) -> tuple[float, float]:
    tx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
    ty = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]
    tw = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(tw) < _W_EPSILON:
        raise ValueError(
            f"homography maps pixel point ({x}, {y}) to the horizon (w={tw}); "
            "not a valid table-plane point"
        )
    return tx / tw, ty / tw
