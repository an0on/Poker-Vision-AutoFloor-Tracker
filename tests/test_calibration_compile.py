"""REQ-9: `calib compile` (`CalibrationAuthoring` -> `CalibrationRuntime`)."""

from __future__ import annotations

import json

import pytest

from poker_vision.calibration.authoring import CalibrationAuthoring
from poker_vision.calibration.compile import compile_calibration
from poker_vision.calibration.geometry import PixelPoint
from poker_vision.calibration.runtime import (
    load_calibration_runtime,
    write_calibration_runtime,
)
from poker_vision.detection.geometry import apply_homography_to_point

VALID_SEATS: list[dict] = [
    {
        "seat_id": "seat_1",
        "zones": {
            "player_area": {
                "points": [
                    {"x": 0, "y": 0},
                    {"x": 100, "y": 0},
                    {"x": 100, "y": 100},
                    {"x": 0, "y": 100},
                ]
            },
            "chip_zone": {
                "points": [
                    {"x": 10, "y": 10},
                    {"x": 50, "y": 10},
                    {"x": 50, "y": 50},
                    {"x": 10, "y": 50},
                ]
            },
        },
    },
    {
        "seat_id": "seat_2",
        "zones": {
            "player_area": {
                "points": [
                    {"x": 200, "y": 0},
                    {"x": 300, "y": 0},
                    {"x": 300, "y": 100},
                    {"x": 200, "y": 100},
                ]
            },
            "chip_zone": {
                "points": [
                    {"x": 210, "y": 10},
                    {"x": 250, "y": 10},
                    {"x": 250, "y": 50},
                    {"x": 210, "y": 50},
                ]
            },
        },
    },
]

VALID_ZONES: dict = {
    "board_zone": {
        "points": [
            {"x": 400, "y": 400},
            {"x": 600, "y": 400},
            {"x": 600, "y": 500},
            {"x": 400, "y": 500},
        ]
    },
    "dealer_area": {
        "points": [
            {"x": 700, "y": 700},
            {"x": 750, "y": 700},
            {"x": 750, "y": 750},
            {"x": 700, "y": 750},
        ]
    },
}

# A plain rectangle-to-rectangle mapping: image (100,100)-(500,400) -> table
# (0,0)-(1000,800), no distortion, so this is easy to hand-verify (scale
# 2.5x in x, 2.0y y, no rotation/skew).
NO_DISTORTION_AUTHORING: dict = {
    "schema_version": "1.0",
    "table_id": "test_table",
    "inference_resolution": {"width": 1920, "height": 1080},
    "camera": {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0},
    "distortion": {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
    "homography": {
        "points": [
            {"image_point": {"x": 100.0, "y": 100.0}, "table_point": {"x": 0.0, "y": 0.0}},
            {"image_point": {"x": 500.0, "y": 100.0}, "table_point": {"x": 1000.0, "y": 0.0}},
            {"image_point": {"x": 500.0, "y": 400.0}, "table_point": {"x": 1000.0, "y": 800.0}},
            {"image_point": {"x": 100.0, "y": 400.0}, "table_point": {"x": 0.0, "y": 800.0}},
        ]
    },
    "table": {"width": 1200.0, "height": 900.0, "unit": "mm"},
    "seats": VALID_SEATS,
    "zones": VALID_ZONES,
}


def _authoring(overrides: dict | None = None) -> CalibrationAuthoring:
    payload = json.loads(json.dumps(NO_DISTORTION_AUTHORING))
    if overrides:
        payload.update(overrides)
    return CalibrationAuthoring.model_validate(payload)


def test_compile_produces_valid_runtime():
    runtime = compile_calibration(_authoring(), based_on="calibration/instance.json")
    assert runtime.schema_version == "1.0"
    assert runtime.table_id == "test_table"
    assert runtime.based_on == "calibration/instance.json"
    assert runtime.inference_resolution.width == 1920
    assert len(runtime.seats) == 2


def test_compile_carries_zones_and_seats_unchanged():
    authoring = _authoring()
    runtime = compile_calibration(authoring, based_on="x.json")
    assert [s.seat_id for s in runtime.seats] == [s.seat_id for s in authoring.seats]
    assert runtime.zones == authoring.zones
    assert runtime.table == authoring.table
    assert runtime.camera == authoring.camera
    assert runtime.distortion == authoring.distortion


def test_compile_solves_homography_matching_hand_computed_scale():
    # No distortion: image (100,100)-(500,400) maps onto table
    # (0,0)-(1000,800) -- a pure axis-aligned scale (2.5x, 2.0y), no
    # rotation/skew/perspective, exactly reproducible by hand.
    runtime = compile_calibration(_authoring(), based_on="x.json")
    center = apply_homography_to_point(
        PixelPoint(x=300.0, y=250.0), runtime.homography, runtime.camera, runtime.distortion
    )
    # x: (300-100)/(500-100)*1000 = 500; y: (250-100)/(400-100)*800 = 400.
    assert center.x == pytest.approx(500.0, abs=1e-6)
    assert center.y == pytest.approx(400.0, abs=1e-6)


def test_compile_homography_round_trips_all_four_correspondences():
    authoring = _authoring()
    runtime = compile_calibration(authoring, based_on="x.json")
    for corr in authoring.homography.points:
        table_point = apply_homography_to_point(
            corr.image_point, runtime.homography, runtime.camera, runtime.distortion
        )
        assert table_point.x == pytest.approx(corr.table_point.x, abs=1e-6)
        assert table_point.y == pytest.approx(corr.table_point.y, abs=1e-6)


def test_compile_output_passes_runtime_schema_validation():
    # `compile_calibration` already returns a validated `CalibrationRuntime`
    # (construction itself runs REQ-11's homography-invertibility check on
    # the *solved* matrix, not the identity stub `test_calibration_schema.py`
    # uses) -- this only pins that the solved matrix is a real 3x3, not a
    # trivial identity/zero fallback.
    runtime = compile_calibration(_authoring(), based_on="x.json")
    assert runtime.homography.forward != [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


# --- AC-6: determinism -------------------------------------------------------


def test_compile_is_deterministic_same_object():
    authoring = _authoring()
    first = compile_calibration(authoring, based_on="x.json")
    second = compile_calibration(authoring, based_on="x.json")
    assert first.model_dump_json() == second.model_dump_json()


def test_compile_is_deterministic_across_reparsed_input(tmp_path):
    # Round-trips the authoring file through disk between the two compiles,
    # closer to what the CLI actually does (load -> compile -> write) than
    # compiling the same in-memory object twice.
    authoring_path = tmp_path / "authoring.json"
    authoring_path.write_text(json.dumps(NO_DISTORTION_AUTHORING))

    from poker_vision.calibration.authoring import load_calibration_authoring

    first = compile_calibration(load_calibration_authoring(authoring_path), based_on="a")
    second = compile_calibration(load_calibration_authoring(authoring_path), based_on="a")
    assert first.model_dump_json() == second.model_dump_json()


def test_write_calibration_runtime_round_trips(tmp_path):
    runtime = compile_calibration(_authoring(), based_on="x.json")
    out_path = tmp_path / "runtime.json"
    write_calibration_runtime(runtime, out_path)
    loaded = load_calibration_runtime(out_path)
    assert loaded.model_dump_json() == runtime.model_dump_json()


def test_write_calibration_runtime_is_byte_identical_across_writes(tmp_path):
    runtime = compile_calibration(_authoring(), based_on="x.json")
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    write_calibration_runtime(runtime, path_a)
    write_calibration_runtime(compile_calibration(_authoring(), based_on="x.json"), path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_compile_rejects_degenerate_homography_points():
    # Four collinear points can't determine a homography.
    payload = json.loads(json.dumps(NO_DISTORTION_AUTHORING))
    payload["homography"]["points"] = [
        {"image_point": {"x": x, "y": 100.0}, "table_point": {"x": x, "y": 0.0}}
        for x in (100.0, 200.0, 300.0, 400.0)
    ]
    authoring = CalibrationAuthoring.model_validate(payload)
    with pytest.raises(ValueError):
        compile_calibration(authoring, based_on="x.json")
