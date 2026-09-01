"""REQ-10: `calib create`'s starting-point authoring skeleton."""

from __future__ import annotations

import pytest

from poker_vision.calibration.geometry import TableUnit
from poker_vision.calibration.skeleton import MIN_SEAT_COUNT, build_authoring_skeleton
from poker_vision.config import Resolution

RESOLUTION = Resolution(width=1920, height=1080)


@pytest.mark.parametrize("seat_count", range(MIN_SEAT_COUNT, MIN_SEAT_COUNT + 8))
def test_skeleton_is_valid_by_construction_for_every_supported_seat_count(seat_count):
    # The whole point of `build_authoring_skeleton`: its output already
    # satisfies REQ-11's zone topology rules without any further editing --
    # construction itself would raise `ValidationError` otherwise.
    authoring = build_authoring_skeleton(
        table_id="t", seat_count=seat_count, table_width=1200.0, table_height=900.0,
        table_unit=TableUnit.MM, inference_resolution=RESOLUTION,
    )
    assert len(authoring.seats) == seat_count
    assert {s.seat_id for s in authoring.seats} == {f"seat_{i + 1}" for i in range(seat_count)}


def test_skeleton_below_min_seat_count_rejected():
    with pytest.raises(ValueError, match=f"seat_count must be >= {MIN_SEAT_COUNT}"):
        build_authoring_skeleton(
            table_id="t", seat_count=MIN_SEAT_COUNT - 1, table_width=1200.0, table_height=900.0,
            table_unit=TableUnit.MM, inference_resolution=RESOLUTION,
        )


def test_skeleton_carries_through_table_id_and_dimensions():
    authoring = build_authoring_skeleton(
        table_id="my_table", seat_count=6, table_width=1500.0, table_height=1000.0,
        table_unit=TableUnit.CM, inference_resolution=RESOLUTION,
    )
    assert authoring.table_id == "my_table"
    assert authoring.table.width == 1500.0
    assert authoring.table.height == 1000.0
    assert authoring.table.unit == TableUnit.CM
    assert authoring.inference_resolution == RESOLUTION


def test_skeleton_homography_has_at_least_four_correspondences():
    authoring = build_authoring_skeleton(
        table_id="t", seat_count=6, table_width=1200.0, table_height=900.0,
        table_unit=TableUnit.MM, inference_resolution=RESOLUTION,
    )
    assert len(authoring.homography.points) >= 4
