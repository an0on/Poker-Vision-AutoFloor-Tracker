"""Pixel-to-table homography (REQ-4, REQ-7, REQ-9).

Authoring stores the raw point correspondences an operator picked (image
pixel <-> table-plane pairs); `calib compile` (REQ-9) solves those into the
precomputed forward/inverse matrices that `CalibrationRuntime` carries, so
runtime code never re-solves a homography on load.
"""

from __future__ import annotations

from pydantic import Field

from poker_vision.calibration.geometry import Matrix3x3, PixelPoint, TablePoint
from poker_vision.schema_base import StrictModel


class HomographyPointCorrespondence(StrictModel):
    image_point: PixelPoint
    table_point: TablePoint


class HomographyCorrespondences(StrictModel):
    """Point pairs used to solve for the pixel -> table-plane homography.

    At least 4 pairs are required (minimum for `cv2.findHomography`/DLT).
    """

    points: list[HomographyPointCorrespondence] = Field(min_length=4)


class HomographyMatrix(StrictModel):
    """Precomputed homography, row-major 3x3.

    `forward` maps undistorted pixel coordinates to table-plane coordinates;
    `inverse` is its matrix inverse (table -> pixel), used e.g. by the debug
    overlay (REQ-37) to draw table-plane geometry back onto the frame.
    """

    forward: Matrix3x3
    inverse: Matrix3x3
