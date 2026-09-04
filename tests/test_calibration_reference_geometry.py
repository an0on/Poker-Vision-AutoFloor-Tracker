"""AC-4 (REQ-6): the reference-photo-derived table geometry is committed and
usable, not just a claim in an ADR. `calibration/authoring/dopo_poker_table.json`
is the canonical, `calib mark-zones`-authored geometry for the physical DOPO
POKER table (see `docs/adr/0001-calibration-geometry-source-reference-photo.md`);
this pins that it actually loads, validates, and compiles.
"""

from __future__ import annotations

from pathlib import Path

from poker_vision.calibration.authoring import (
    CALIBRATION_AUTHORING_SCHEMA_VERSION,
    load_calibration_authoring,
)
from poker_vision.calibration.compile import compile_calibration

REFERENCE_AUTHORING_PATH = (
    Path(__file__).resolve().parent.parent / "calibration" / "authoring" / "dopo_poker_table.json"
)


def test_reference_authoring_file_exists():
    assert REFERENCE_AUTHORING_PATH.is_file(), (
        f"canonical calibration authoring geometry missing at {REFERENCE_AUTHORING_PATH}"
    )


def test_reference_authoring_loads_and_validates():
    authoring = load_calibration_authoring(REFERENCE_AUTHORING_PATH)
    assert authoring.schema_version == CALIBRATION_AUTHORING_SCHEMA_VERSION
    assert len(authoring.seats) == 10
    assert {seat.seat_id for seat in authoring.seats} == {f"seat_{i}" for i in range(1, 11)}


def test_reference_authoring_card_dealer_seat_is_one_of_the_ten_seats():
    authoring = load_calibration_authoring(REFERENCE_AUTHORING_PATH)
    assert authoring.card_dealer_seat_id in {seat.seat_id for seat in authoring.seats}


def test_reference_authoring_compiles_to_a_req11_valid_runtime():
    authoring = load_calibration_authoring(REFERENCE_AUTHORING_PATH)
    # `compile_calibration` returning at all already proves REQ-11 validity:
    # `CalibrationRuntime`'s own validators (polygon/zone/homography checks)
    # run during construction and raise on any violation.
    runtime = compile_calibration(authoring, based_on=str(REFERENCE_AUTHORING_PATH))
    assert len(runtime.seats) == 10
    assert runtime.card_dealer_seat_id == authoring.card_dealer_seat_id
