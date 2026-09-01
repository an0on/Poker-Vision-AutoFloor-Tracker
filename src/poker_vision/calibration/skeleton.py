"""`calib create`'s starting-point authoring skeleton (REQ-10).

Generates a `CalibrationAuthoring` that is valid by construction -- every
zone already satisfies REQ-11's topology rules -- for an operator to then
adjust seat-by-seat with `calib edit` (or by hand) to match their actual
physical table and camera view. Not the real v3-landscape table geometry
(REQ-6 migrates that separately, out of scope here): seats are laid out as
even wedges around an ellipse inscribed in the table rectangle, which is
just a reasonable, always-valid default, not a claim about any specific
physical table.
"""

from __future__ import annotations

import math

from poker_vision.calibration.authoring import (
    CALIBRATION_AUTHORING_SCHEMA_VERSION,
    CalibrationAuthoring,
)
from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import (
    TablePoint,
    TablePolygon,
    TableUnit,
    polygon_centroid,
)
from poker_vision.calibration.homography import (
    HomographyCorrespondences,
    HomographyPointCorrespondence,
)
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.config import Resolution

MIN_SEAT_COUNT = 3
"""Below 3 seats, this ellipse-wedge layout can produce a degenerate (zero-area,
collinear) wedge -- see this module's own derivation in the compile CLI's
tests. A real 2-max table is representable, just not by this generator;
author it by hand or via `calib edit` from a 3-seat skeleton instead."""

# Fractions of table width/height defining the wedge annulus and the central
# board/dealer zones; chosen with generous clearance between them so any
# seat count >= MIN_SEAT_COUNT validates against REQ-11 by construction.
_OUTER_RADIUS_FRACTION = 0.48
_INNER_RADIUS_FRACTION = 0.30
_CHIP_ZONE_SHRINK_FACTOR = 0.5  # chip_zone = player_area scaled toward its own centroid
_BOARD_ZONE_HALF_EXTENT_FRACTION = 0.10
_DEALER_ZONE_HALF_EXTENT_FRACTION = 0.03
_DEALER_ZONE_CENTER_OFFSET_FRACTION = 0.20
_HOMOGRAPHY_IMAGE_MARGIN_FRACTION = 0.10

_DEFAULT_FX = 1400.0
_DEFAULT_FY = 1400.0


def _ellipse_point(center: TablePoint, rx: float, ry: float, angle: float) -> TablePoint:
    return TablePoint(x=center.x + rx * math.cos(angle), y=center.y + ry * math.sin(angle))


def _scale_toward_centroid(polygon: TablePolygon, factor: float) -> TablePolygon:
    centroid = polygon_centroid(polygon)
    return TablePolygon(
        points=[
            TablePoint(
                x=centroid.x + factor * (p.x - centroid.x),
                y=centroid.y + factor * (p.y - centroid.y),
            )
            for p in polygon.points
        ]
    )


def _build_seat(
    seat_id: str,
    center: TablePoint,
    width: float,
    height: float,
    index: int,
    seat_count: int,
) -> CalibrationSeat:
    start_angle = 2.0 * math.pi * index / seat_count
    end_angle = 2.0 * math.pi * (index + 1) / seat_count
    rx_outer, ry_outer = width * _OUTER_RADIUS_FRACTION, height * _OUTER_RADIUS_FRACTION
    rx_inner, ry_inner = width * _INNER_RADIUS_FRACTION, height * _INNER_RADIUS_FRACTION

    player_area = TablePolygon(
        points=[
            _ellipse_point(center, rx_outer, ry_outer, start_angle),
            _ellipse_point(center, rx_outer, ry_outer, end_angle),
            _ellipse_point(center, rx_inner, ry_inner, end_angle),
            _ellipse_point(center, rx_inner, ry_inner, start_angle),
        ]
    )
    chip_zone = _scale_toward_centroid(player_area, _CHIP_ZONE_SHRINK_FACTOR)
    return CalibrationSeat(
        seat_id=seat_id, zones=SeatZones(player_area=player_area, chip_zone=chip_zone)
    )


def _axis_aligned_square(
    center: TablePoint, half_extent_x: float, half_extent_y: float
) -> TablePolygon:
    return TablePolygon(
        points=[
            TablePoint(x=center.x - half_extent_x, y=center.y - half_extent_y),
            TablePoint(x=center.x + half_extent_x, y=center.y - half_extent_y),
            TablePoint(x=center.x + half_extent_x, y=center.y + half_extent_y),
            TablePoint(x=center.x - half_extent_x, y=center.y + half_extent_y),
        ]
    )


def build_authoring_skeleton(
    table_id: str,
    seat_count: int,
    table_width: float,
    table_height: float,
    table_unit: TableUnit,
    inference_resolution: Resolution,
) -> CalibrationAuthoring:
    """Build a REQ-11-valid `CalibrationAuthoring` skeleton for `calib create`.

    Raises `ValueError` if `seat_count < MIN_SEAT_COUNT` (see that constant's
    docstring for why fewer seats can't be laid out this way without risking
    a degenerate wedge).
    """
    if seat_count < MIN_SEAT_COUNT:
        raise ValueError(
            f"seat_count must be >= {MIN_SEAT_COUNT} for this layout "
            f"(got {seat_count}); author fewer seats by hand or via `calib edit`"
        )

    center = TablePoint(x=table_width / 2.0, y=table_height / 2.0)
    seats = [
        _build_seat(f"seat_{i + 1}", center, table_width, table_height, i, seat_count)
        for i in range(seat_count)
    ]

    board_zone = _axis_aligned_square(
        center,
        table_width * _BOARD_ZONE_HALF_EXTENT_FRACTION,
        table_height * _BOARD_ZONE_HALF_EXTENT_FRACTION,
    )
    dealer_center = TablePoint(
        x=center.x + table_width * _DEALER_ZONE_CENTER_OFFSET_FRACTION,
        y=center.y - table_height * _DEALER_ZONE_CENTER_OFFSET_FRACTION,
    )
    dealer_area = _axis_aligned_square(
        dealer_center,
        table_width * _DEALER_ZONE_HALF_EXTENT_FRACTION,
        table_height * _DEALER_ZONE_HALF_EXTENT_FRACTION,
    )

    margin_x = inference_resolution.width * _HOMOGRAPHY_IMAGE_MARGIN_FRACTION
    margin_y = inference_resolution.height * _HOMOGRAPHY_IMAGE_MARGIN_FRACTION
    image_corners = [
        (margin_x, margin_y),
        (inference_resolution.width - margin_x, margin_y),
        (inference_resolution.width - margin_x, inference_resolution.height - margin_y),
        (margin_x, inference_resolution.height - margin_y),
    ]
    table_corners = [
        (0.0, 0.0),
        (table_width, 0.0),
        (table_width, table_height),
        (0.0, table_height),
    ]
    homography = HomographyCorrespondences(
        points=[
            HomographyPointCorrespondence(
                image_point={"x": ix, "y": iy}, table_point={"x": tx, "y": ty}
            )
            for (ix, iy), (tx, ty) in zip(image_corners, table_corners, strict=True)
        ]
    )

    return CalibrationAuthoring(
        schema_version=CALIBRATION_AUTHORING_SCHEMA_VERSION,
        table_id=table_id,
        inference_resolution=inference_resolution,
        camera=CameraIntrinsics(
            fx=_DEFAULT_FX,
            fy=_DEFAULT_FY,
            cx=inference_resolution.width / 2.0,
            cy=inference_resolution.height / 2.0,
        ),
        distortion=DistortionCoefficients(),
        homography=homography,
        table={"width": table_width, "height": table_height, "unit": table_unit},
        seats=seats,
        zones=GlobalZones(board_zone=board_zone, dealer_area=dealer_area),
        # Arbitrary but always valid: seat_1 always exists (seat_count >= MIN_SEAT_COUNT).
        # This skeleton generator has no notion of a physical Kartengeber position -- the
        # real one is authored by `calib mark-zones` (REQ-10a) against the reference photo.
        card_dealer_seat_id=seats[0].seat_id,
    )
