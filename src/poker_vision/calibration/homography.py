"""Pixel-to-table homography (REQ-4, REQ-7, REQ-9).

Authoring stores the raw point correspondences an operator picked (image
pixel <-> table-plane pairs); `calib compile` (REQ-9) solves those into the
precomputed forward/inverse matrices that `CalibrationRuntime` carries, so
runtime code never re-solves a homography on load.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from poker_vision.calibration.geometry import (
    Matrix3x3,
    PixelPoint,
    TablePoint,
    matrix3x3_determinant,
    matrix3x3_multiply,
)
from poker_vision.schema_base import StrictModel

# Determinant magnitude below this is treated as singular (REQ-11).
_DETERMINANT_EPSILON = 1e-9
# Tolerance for forward @ inverse == identity (REQ-11); generous relative to
# double-precision round-trip error, tight enough to catch a wrong/stale
# inverse.
_IDENTITY_EPSILON = 1e-6


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

    @model_validator(mode="after")
    def _check_invertible(self) -> HomographyMatrix:
        # REQ-11: a singular forward matrix has no real inverse, so whatever
        # is stored in `inverse` would be meaningless.
        if abs(matrix3x3_determinant(self.forward)) < _DETERMINANT_EPSILON:
            raise ValueError("homography is not invertible: forward matrix has zero determinant")

        # REQ-11 also covers the stored `inverse` itself being wrong/stale
        # (e.g. left over from a different `forward`) — pipeline code trusts
        # it as-is (see class docstring) rather than re-solving it, so a
        # mismatch here would silently corrupt the debug overlay.
        product = matrix3x3_multiply(self.forward, self.inverse)
        for row in range(3):
            for col in range(3):
                expected = 1.0 if row == col else 0.0
                if abs(product[row][col] - expected) > _IDENTITY_EPSILON:
                    raise ValueError(
                        "homography is not invertible: inverse does not satisfy "
                        "forward @ inverse = identity"
                    )
        return self
