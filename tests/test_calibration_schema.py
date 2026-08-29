import json

import pytest
from pydantic import ValidationError

from poker_vision.calibration.authoring import CalibrationAuthoring, load_calibration_authoring
from poker_vision.calibration.runtime import CalibrationRuntime, load_calibration_runtime

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

VALID_CAMERA: dict = {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0}
VALID_DISTORTION: dict = {"k1": 0.01, "k2": -0.02, "p1": 0.0, "p2": 0.0, "k3": 0.0}

VALID_AUTHORING: dict = {
    "schema_version": "1.0",
    "table_id": "test_table",
    "image": {"width": 1920, "height": 1080},
    "camera": VALID_CAMERA,
    "distortion": VALID_DISTORTION,
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

IDENTITY_MATRIX = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

VALID_RUNTIME: dict = {
    "schema_version": "1.0",
    "table_id": "test_table",
    "based_on": "calibration/instance.json",
    "image": {"width": 1920, "height": 1080},
    "camera": VALID_CAMERA,
    "distortion": VALID_DISTORTION,
    "homography": {"forward": IDENTITY_MATRIX, "inverse": IDENTITY_MATRIX},
    "table": {"width": 1200.0, "height": 900.0, "unit": "mm"},
    "seats": VALID_SEATS,
    "zones": VALID_ZONES,
}


def _payload(base: dict, **overrides: object) -> dict:
    merged = json.loads(json.dumps(base))
    for key, value in overrides.items():
        merged[key] = value
    return merged


# --- Authoring -----------------------------------------------------------


def test_valid_authoring_loads():
    calibration = CalibrationAuthoring.model_validate(VALID_AUTHORING)
    assert calibration.schema_version == "1.0"
    assert len(calibration.seats) == 2
    assert calibration.seats[0].seat_id == "seat_1"
    assert calibration.table.unit.value == "mm"


# AC-3: wrong schema_version fails
def test_authoring_wrong_schema_version_rejected():
    with pytest.raises(ValidationError):
        CalibrationAuthoring.model_validate(_payload(VALID_AUTHORING, schema_version="2.0"))


def test_authoring_missing_schema_version_rejected():
    payload = json.loads(json.dumps(VALID_AUTHORING))
    del payload["schema_version"]
    with pytest.raises(ValidationError):
        CalibrationAuthoring.model_validate(payload)


# AC-3: unknown top-level field fails
def test_authoring_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        CalibrationAuthoring.model_validate(_payload(VALID_AUTHORING, unexpected_field="nope"))


# AC-3: unknown nested field fails
def test_authoring_unknown_nested_field_rejected():
    payload = _payload(VALID_AUTHORING)
    payload["camera"] = {**VALID_CAMERA, "typo_field": 1}
    with pytest.raises(ValidationError):
        CalibrationAuthoring.model_validate(payload)


def test_authoring_duplicate_seat_id_rejected():
    payload = _payload(VALID_AUTHORING)
    payload["seats"][1]["seat_id"] = "seat_1"
    with pytest.raises(ValidationError, match="unique"):
        CalibrationAuthoring.model_validate(payload)


def test_authoring_empty_seats_rejected():
    with pytest.raises(ValidationError):
        CalibrationAuthoring.model_validate(_payload(VALID_AUTHORING, seats=[]))


def test_authoring_polygon_needs_at_least_three_points():
    payload = _payload(VALID_AUTHORING)
    payload["zones"]["board_zone"]["points"] = [{"x": 0, "y": 0}, {"x": 1, "y": 1}]
    with pytest.raises(ValidationError):
        CalibrationAuthoring.model_validate(payload)


def test_authoring_homography_needs_at_least_four_points():
    payload = _payload(VALID_AUTHORING)
    payload["homography"]["points"] = payload["homography"]["points"][:3]
    with pytest.raises(ValidationError):
        CalibrationAuthoring.model_validate(payload)


def test_load_calibration_authoring_from_json_file(tmp_path):
    path = tmp_path / "authoring.json"
    path.write_text(json.dumps(VALID_AUTHORING))
    calibration = load_calibration_authoring(path)
    assert calibration.schema_version == "1.0"


def test_load_calibration_authoring_invalid_json_raises(tmp_path):
    path = tmp_path / "authoring.json"
    path.write_text("{not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_calibration_authoring(path)


def test_load_calibration_authoring_schema_violation_raises(tmp_path):
    path = tmp_path / "authoring.json"
    path.write_text(json.dumps(_payload(VALID_AUTHORING, schema_version="2.0")))
    with pytest.raises(ValueError, match="invalid calibration"):
        load_calibration_authoring(path)


# --- Runtime ---------------------------------------------------------------


def test_valid_runtime_loads():
    calibration = CalibrationRuntime.model_validate(VALID_RUNTIME)
    assert calibration.schema_version == "1.0"
    assert calibration.homography.forward == IDENTITY_MATRIX
    assert calibration.homography.inverse == IDENTITY_MATRIX


def test_runtime_wrong_schema_version_rejected():
    with pytest.raises(ValidationError):
        CalibrationRuntime.model_validate(_payload(VALID_RUNTIME, schema_version="2.0"))


def test_runtime_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        CalibrationRuntime.model_validate(_payload(VALID_RUNTIME, unexpected_field="nope"))


def test_runtime_unknown_nested_field_rejected():
    payload = _payload(VALID_RUNTIME)
    payload["homography"] = {**payload["homography"], "typo_field": 1}
    with pytest.raises(ValidationError):
        CalibrationRuntime.model_validate(payload)


def test_runtime_duplicate_seat_id_rejected():
    payload = _payload(VALID_RUNTIME)
    payload["seats"][1]["seat_id"] = "seat_1"
    with pytest.raises(ValidationError, match="unique"):
        CalibrationRuntime.model_validate(payload)


@pytest.mark.parametrize(
    "bad_matrix",
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    ],
)
def test_runtime_homography_matrix_must_be_3x3(bad_matrix):
    payload = _payload(VALID_RUNTIME)
    payload["homography"]["forward"] = bad_matrix
    with pytest.raises(ValidationError):
        CalibrationRuntime.model_validate(payload)


def test_load_calibration_runtime_from_json_file(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(VALID_RUNTIME))
    calibration = load_calibration_runtime(path)
    assert calibration.schema_version == "1.0"


def test_load_calibration_runtime_schema_violation_raises(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(_payload(VALID_RUNTIME, schema_version="2.0")))
    with pytest.raises(ValueError, match="invalid calibration"):
        load_calibration_runtime(path)


# --- REQ-11: hard geometric validation on load ------------------------------
#
# AC-7: every rule listed in REQ-11 gets its own test, checked against both
# schemas where the rule applies to both (they share the validation code via
# `CalibrationGeometryModel`/`TablePolygon`); homography invertibility only
# applies to CalibrationRuntime, since CalibrationAuthoring stores raw point
# correspondences rather than a solved matrix.

DEGENERATE_COLLINEAR_POINTS = [
    {"x": 400, "y": 400},
    {"x": 500, "y": 400},
    {"x": 600, "y": 400},
]
DEGENERATE_COINCIDENT_POINTS = [
    {"x": 400, "y": 400},
    {"x": 400, "y": 400},
    {"x": 400, "y": 400},
]

CHIP_ZONE_OUTSIDE_SEAT_1_PLAYER_AREA: dict = {
    "points": [
        {"x": 150, "y": 10},
        {"x": 190, "y": 10},
        {"x": 190, "y": 50},
        {"x": 150, "y": 50},
    ]
}

# Two seats whose chip_zones overlap each other while each individually
# stays inside its own player_area (so this isolates the cross-seat overlap
# rule from the containment rule). Both chip_zones share the same y-range,
# which is exactly the "flush edge" layout `polygons_overlap` has to detect
# via more than just vertex-in-polygon checks.
SEATS_WITH_OVERLAPPING_CHIP_ZONES: list[dict] = [
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
                    {"x": 40, "y": 10},
                    {"x": 90, "y": 10},
                    {"x": 90, "y": 50},
                    {"x": 40, "y": 50},
                ]
            },
        },
    },
    {
        "seat_id": "seat_2",
        "zones": {
            "player_area": {
                "points": [
                    {"x": 50, "y": 0},
                    {"x": 150, "y": 0},
                    {"x": 150, "y": 100},
                    {"x": 50, "y": 100},
                ]
            },
            "chip_zone": {
                "points": [
                    {"x": 60, "y": 10},
                    {"x": 110, "y": 10},
                    {"x": 110, "y": 50},
                    {"x": 60, "y": 50},
                ]
            },
        },
    },
]

BOARD_ZONE_OVERLAPPING_SEAT_1_CHIP_ZONE: dict = {
    "points": [
        {"x": 30, "y": 30},
        {"x": 70, "y": 30},
        {"x": 70, "y": 70},
        {"x": 30, "y": 70},
    ]
}


@pytest.mark.parametrize(
    "degenerate_points", [DEGENERATE_COLLINEAR_POINTS, DEGENERATE_COINCIDENT_POINTS]
)
def test_authoring_degenerate_polygon_rejected(degenerate_points):
    payload = _payload(VALID_AUTHORING)
    payload["zones"]["board_zone"]["points"] = degenerate_points
    with pytest.raises(ValidationError, match="degenerate"):
        CalibrationAuthoring.model_validate(payload)


def test_runtime_degenerate_polygon_rejected():
    payload = _payload(VALID_RUNTIME)
    payload["zones"]["board_zone"]["points"] = DEGENERATE_COLLINEAR_POINTS
    with pytest.raises(ValidationError, match="degenerate"):
        CalibrationRuntime.model_validate(payload)


def test_authoring_chip_zone_outside_player_area_rejected():
    payload = _payload(VALID_AUTHORING)
    payload["seats"][0]["zones"]["chip_zone"] = CHIP_ZONE_OUTSIDE_SEAT_1_PLAYER_AREA
    with pytest.raises(ValidationError, match="not fully contained"):
        CalibrationAuthoring.model_validate(payload)


def test_runtime_chip_zone_outside_player_area_rejected():
    payload = _payload(VALID_RUNTIME)
    payload["seats"][0]["zones"]["chip_zone"] = CHIP_ZONE_OUTSIDE_SEAT_1_PLAYER_AREA
    with pytest.raises(ValidationError, match="not fully contained"):
        CalibrationRuntime.model_validate(payload)


def test_authoring_chip_zone_overlap_between_seats_rejected():
    payload = _payload(VALID_AUTHORING, seats=SEATS_WITH_OVERLAPPING_CHIP_ZONES)
    with pytest.raises(ValidationError, match="chip_zone overlap between seats"):
        CalibrationAuthoring.model_validate(payload)


def test_runtime_chip_zone_overlap_between_seats_rejected():
    payload = _payload(VALID_RUNTIME, seats=SEATS_WITH_OVERLAPPING_CHIP_ZONES)
    with pytest.raises(ValidationError, match="chip_zone overlap between seats"):
        CalibrationRuntime.model_validate(payload)


def test_runtime_identical_chip_zone_copy_pasted_between_seats_rejected():
    # Realistic authoring mistake: seat_2's chip_zone block copy-pasted from
    # seat_1 without updating the coordinates. Uses SEATS_WITH_OVERLAPPING_
    # CHIP_ZONES' player_areas (which overlap in x:[50, 100]) so one
    # rectangle can validly sit inside *both* player_areas in isolation.
    # Every vertex/edge-midpoint of each chip_zone then lands exactly on the
    # other's boundary rather than strictly inside it, and every edge pair
    # is collinear rather than crossing — the most extreme case of overlap,
    # but also the easiest for a naive vertex/crossing-only check to miss.
    shared_chip_zone = {
        "points": [
            {"x": 60, "y": 10},
            {"x": 90, "y": 10},
            {"x": 90, "y": 50},
            {"x": 60, "y": 50},
        ]
    }
    payload = _payload(VALID_RUNTIME, seats=SEATS_WITH_OVERLAPPING_CHIP_ZONES)
    payload["seats"][0]["zones"]["chip_zone"] = shared_chip_zone
    payload["seats"][1]["zones"]["chip_zone"] = shared_chip_zone
    with pytest.raises(ValidationError, match="chip_zone overlap between seats"):
        CalibrationRuntime.model_validate(payload)


def test_authoring_board_zone_overlaps_chip_zone_rejected():
    payload = _payload(VALID_AUTHORING)
    payload["zones"]["board_zone"] = BOARD_ZONE_OVERLAPPING_SEAT_1_CHIP_ZONE
    with pytest.raises(ValidationError, match="board_zone overlaps chip_zone"):
        CalibrationAuthoring.model_validate(payload)


def test_runtime_board_zone_overlaps_chip_zone_rejected():
    payload = _payload(VALID_RUNTIME)
    payload["zones"]["board_zone"] = BOARD_ZONE_OVERLAPPING_SEAT_1_CHIP_ZONE
    with pytest.raises(ValidationError, match="board_zone overlaps chip_zone"):
        CalibrationRuntime.model_validate(payload)


def test_runtime_homography_singular_forward_rejected():
    payload = _payload(VALID_RUNTIME)
    payload["homography"]["forward"] = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    # `inverse` is left as the identity (from VALID_RUNTIME): a singular
    # forward can never round-trip with *any* inverse, so this is rejected
    # regardless of what `inverse` happens to hold.
    with pytest.raises(ValidationError, match="does not equal the identity matrix"):
        CalibrationRuntime.model_validate(payload)


def test_runtime_homography_inverse_inconsistent_with_forward_rejected():
    payload = _payload(VALID_RUNTIME)
    # `forward` is a perfectly good (identity) matrix, but this `inverse` is
    # not *its* inverse (identity, not 2 * identity) — a stale/wrong value.
    payload["homography"]["inverse"] = [
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ]
    with pytest.raises(ValidationError, match="does not equal the identity matrix"):
        CalibrationRuntime.model_validate(payload)


def test_runtime_homography_small_scale_still_accepted():
    # A homography is only defined up to a nonzero scalar; a small-but-valid
    # scaling must not be rejected as "singular" (regression test for a
    # since-removed absolute-determinant check, which was scale-sensitive:
    # det of a 3x3 matrix scales with the cube of the scaling factor, so
    # det(0.0001 * I) = 1e-12 — far below what any fixed epsilon would call
    # "not singular" even though this pair round-trips exactly).
    payload = _payload(VALID_RUNTIME)
    payload["homography"] = {
        "forward": [[0.0001, 0.0, 0.0], [0.0, 0.0001, 0.0], [0.0, 0.0, 0.0001]],
        "inverse": [[10000.0, 0.0, 0.0], [0.0, 10000.0, 0.0], [0.0, 0.0, 10000.0]],
    }
    calibration = CalibrationRuntime.model_validate(payload)
    assert calibration.homography.forward[0][0] == 0.0001


def test_runtime_homography_overflow_to_nan_rejected():
    # Individually-finite entries (each passes Matrix3x3's own finite-value
    # check) can still overflow when multiplied: forward[0][0]*inverse[0][0]
    # and forward[0][1]*inverse[1][0] each overflow to +-inf, and their sum
    # in the same dot product collapses to NaN. `abs(nan - expected) >
    # epsilon` is always False, so this must be caught by an explicit
    # finiteness check on the product, not the identity comparison alone.
    payload = _payload(VALID_RUNTIME)
    payload["homography"] = {
        "forward": [[1e200, -1e200, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "inverse": [[1e200, 0.0, 0.0], [1e200, 0.0, 0.0], [0.0, 0.0, 1.0]],
    }
    with pytest.raises(ValidationError, match="non-finite value"):
        CalibrationRuntime.model_validate(payload)


SELF_INTERSECTING_BOWTIE_POINTS = [
    {"x": 400, "y": 400},
    {"x": 404, "y": 404},
    {"x": 404, "y": 400},
    {"x": 400, "y": 410},
]


def test_authoring_self_intersecting_polygon_rejected():
    payload = _payload(VALID_AUTHORING)
    payload["zones"]["board_zone"]["points"] = SELF_INTERSECTING_BOWTIE_POINTS
    with pytest.raises(ValidationError, match="self-intersect"):
        CalibrationAuthoring.model_validate(payload)


def test_runtime_self_intersecting_polygon_rejected():
    payload = _payload(VALID_RUNTIME)
    payload["zones"]["board_zone"]["points"] = SELF_INTERSECTING_BOWTIE_POINTS
    with pytest.raises(ValidationError, match="self-intersect"):
        CalibrationRuntime.model_validate(payload)


# NaN/inf would otherwise silently defeat the checks above: e.g.
# `abs(nan) < epsilon` and `abs(nan - expected) > epsilon` are both always
# False in Python/IEEE 754, so a NaN coordinate or matrix entry would sail
# through the degenerate-area and homography-invertibility checks. Rejected
# at the type level instead (TablePoint/PixelPoint, Matrix3x3).


def test_authoring_polygon_nan_coordinate_rejected():
    payload = _payload(VALID_AUTHORING)
    payload["zones"]["board_zone"]["points"][0]["x"] = float("nan")
    with pytest.raises(ValidationError):
        CalibrationAuthoring.model_validate(payload)


def test_runtime_polygon_nan_coordinate_rejected():
    payload = _payload(VALID_RUNTIME)
    payload["zones"]["board_zone"]["points"][0]["x"] = float("nan")
    with pytest.raises(ValidationError):
        CalibrationRuntime.model_validate(payload)


def test_runtime_homography_nan_entry_rejected():
    payload = _payload(VALID_RUNTIME)
    payload["homography"]["forward"][0][0] = float("nan")
    with pytest.raises(ValidationError):
        CalibrationRuntime.model_validate(payload)


def test_load_calibration_runtime_with_nan_coordinate_from_json_file_rejected(tmp_path):
    # End-to-end: json.loads accepts the non-standard (RFC 8259 forbids it)
    # `NaN` literal by default, and json.dumps emits it for float("nan") the
    # same way — so this exercises the exact loader path a hand-edited or
    # buggy authoring file could hit, not just direct model construction.
    payload = _payload(VALID_RUNTIME)
    payload["zones"]["board_zone"]["points"][0]["x"] = float("nan")
    raw = json.dumps(payload)
    assert "NaN" in raw
    path = tmp_path / "runtime.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="invalid calibration"):
        load_calibration_runtime(path)
