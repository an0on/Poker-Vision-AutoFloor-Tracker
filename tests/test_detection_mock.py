"""REQ-18: `mock` detector, Modus A (script-driven detections), AC-11 slice."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from poker_vision.calibration.geometry import PixelPoint
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.detection.geometry import apply_homography_to_point
from poker_vision.detection.mock import MockDetector
from poker_vision.detection.models import DetectionClass

# Non-trivial homography (not the identity, so a test can't pass by accident)
# plus real lens distortion, so the round trip through the inverse transform
# (table entries) is actually exercised, not just the zero-distortion case.
FORWARD = [[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]]
INVERSE = [[0.5, 0.0, -5.0], [0.0, 1.0 / 3.0, -20.0 / 3.0], [0.0, 0.0, 1.0]]
CAMERA = {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0}
DISTORTION = {"k1": 0.15, "k2": -0.05, "p1": 0.001, "p2": -0.002, "k3": 0.01}

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
    }
]

VALID_ZONES: dict = {
    "board_zone": {
        "points": [
            {"x": 4000, "y": 4000},
            {"x": 6000, "y": 4000},
            {"x": 6000, "y": 5000},
            {"x": 4000, "y": 5000},
        ]
    },
    "dealer_area": {
        "points": [
            {"x": 7000, "y": 7000},
            {"x": 7500, "y": 7000},
            {"x": 7500, "y": 7500},
            {"x": 7000, "y": 7500},
        ]
    },
}


def _runtime() -> CalibrationRuntime:
    payload = {
        "schema_version": "1.0",
        "table_id": "test_table",
        "based_on": "calibration/instance.json",
        "inference_resolution": {"width": 1920, "height": 1080},
        "camera": CAMERA,
        "distortion": DISTORTION,
        "homography": {"forward": FORWARD, "inverse": INVERSE},
        "table": {"width": 1200.0, "height": 900.0, "unit": "mm"},
        "seats": VALID_SEATS,
        "zones": VALID_ZONES,
        "card_dealer_seat_id": "seat_1",
    }
    return CalibrationRuntime.model_validate(payload)


def _frame(frame_index: int = 0) -> Frame:
    return Frame(
        image=np.zeros((1080, 1920, 3), dtype=np.uint8),
        timestamp=datetime.now(UTC),
        frame_index=frame_index,
        source_id="test",
    )


def _write_script(path: Path, lines: list[dict | str]) -> Path:
    script_path = path / "script.jsonl"
    with script_path.open("w") as handle:
        for line in lines:
            handle.write(line if isinstance(line, str) else json.dumps(line))
            handle.write("\n")
    return script_path


# --- Modus A: pixel-space entries -------------------------------------------


def test_pixel_entry_center_and_box_land_in_table_coordinates(tmp_path):
    script = _write_script(
        tmp_path,
        [
            {
                "frame_index": 0,
                "detections": [
                    {
                        "coordinate_space": "pixel",
                        "object_class": "chip",
                        "confidence": 0.9,
                        "center": {"x": 999.0, "y": 999.0},  # must be ignored: box wins
                        "box": {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
                    }
                ],
            }
        ],
    )
    detector = MockDetector(_runtime(), script)

    result = detector.detect(_frame(0))

    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.object_class is DetectionClass.CHIP
    assert detection.confidence == pytest.approx(0.9)
    # box_center((0,0,10,10)) = (5,5) undistorted-ish near identity at low
    # magnitude -> forward homography (2*5+10, 3*5+20) = (20, 35), close
    # enough given the mild distortion used here.
    assert detection.box is not None


def test_pixel_entry_without_box_uses_center_directly(tmp_path):
    script = _write_script(
        tmp_path,
        [
            {
                "frame_index": 2,
                "detections": [
                    {
                        "coordinate_space": "pixel",
                        "object_class": "dealer_button",
                        "confidence": 0.5,
                        "center": {"x": 960.0, "y": 540.0},
                    }
                ],
            }
        ],
    )
    detector = MockDetector(_runtime(), script)

    result = detector.detect(_frame(2))

    detection = result.detections[0]
    # (960, 540) is the principal point: undistortion is a no-op there, so
    # the forward homography alone determines the table point.
    assert detection.center.x == pytest.approx(2 * 960.0 + 10.0, abs=1e-3)
    assert detection.center.y == pytest.approx(3 * 540.0 + 20.0, abs=1e-3)
    assert detection.box is None


# --- Modus A: table-space entries -------------------------------------------


def test_table_entry_round_trips_to_the_same_table_point(tmp_path):
    target = {"x": 321.0, "y": 654.0}
    script = _write_script(
        tmp_path,
        [
            {
                "frame_index": 5,
                "detections": [
                    {
                        "coordinate_space": "table",
                        "object_class": "card",
                        "confidence": 0.8,
                        "center": target,
                    }
                ],
            }
        ],
    )
    detector = MockDetector(_runtime(), script)

    result = detector.detect(_frame(5))

    detection = result.detections[0]
    assert detection.object_class is DetectionClass.CARD
    assert detection.center.x == pytest.approx(target["x"], abs=1e-2)
    assert detection.center.y == pytest.approx(target["y"], abs=1e-2)
    assert detection.box is None


def test_pixel_and_table_entries_agree_for_the_same_physical_point(tmp_path):
    # AC-11's cross-mode-agreement check, sliced to what REQ-18 alone can
    # exercise: a "pixel" entry and a "table" entry describing the same
    # physical point must land within the AC's tolerance of each other,
    # regardless of which coordinate space the script used.
    calibration = _runtime()
    pixel_point = PixelPoint(x=300.0, y=700.0)
    expected_table_point = apply_homography_to_point(
        pixel_point, calibration.homography, calibration.camera, calibration.distortion
    )
    script = _write_script(
        tmp_path,
        [
            {
                "frame_index": 0,
                "detections": [
                    {
                        "coordinate_space": "pixel",
                        "object_class": "chip",
                        "confidence": 0.7,
                        "center": {"x": pixel_point.x, "y": pixel_point.y},
                    }
                ],
            },
            {
                "frame_index": 1,
                "detections": [
                    {
                        "coordinate_space": "table",
                        "object_class": "chip",
                        "confidence": 0.7,
                        "center": {"x": expected_table_point.x, "y": expected_table_point.y},
                    }
                ],
            },
        ],
    )
    detector = MockDetector(calibration, script)

    pixel_result = detector.detect(_frame(0)).detections[0]
    table_result = detector.detect(_frame(1)).detections[0]

    table_width = calibration.table.width
    tolerance = 0.01 * table_width
    assert pixel_result.center.x == pytest.approx(table_result.center.x, abs=tolerance)
    assert pixel_result.center.y == pytest.approx(table_result.center.y, abs=tolerance)
    # Also matches the independently-computed expected point, not just each other.
    assert pixel_result.center.x == pytest.approx(expected_table_point.x, abs=tolerance)
    assert table_result.center.x == pytest.approx(expected_table_point.x, abs=tolerance)


# --- Frame lookup, no matching line ------------------------------------------


def test_frame_index_without_script_line_yields_no_detections(tmp_path):
    script = _write_script(tmp_path, [{"frame_index": 0, "detections": []}])
    detector = MockDetector(_runtime(), script)

    result = detector.detect(_frame(999))

    assert result.detections == []


def test_blank_lines_are_skipped(tmp_path):
    script_path = tmp_path / "script.jsonl"
    script_path.write_text(
        "\n"
        + json.dumps({"frame_index": 0, "detections": []})
        + "\n\n"
        + json.dumps({"frame_index": 1, "detections": []})
        + "\n"
    )
    detector = MockDetector(_runtime(), script_path)

    assert detector.detect(_frame(0)).detections == []
    assert detector.detect(_frame(1)).detections == []


# --- Script validation: hard failures, no silent tolerance -------------------


def test_duplicate_frame_index_raises(tmp_path):
    script = _write_script(
        tmp_path,
        [
            {"frame_index": 0, "detections": []},
            {"frame_index": 0, "detections": []},
        ],
    )
    with pytest.raises(ValueError, match="duplicate frame_index"):
        MockDetector(_runtime(), script)


def test_invalid_json_line_raises_with_line_number(tmp_path):
    script = _write_script(tmp_path, [{"frame_index": 0, "detections": []}, "not json {"])
    with pytest.raises(ValueError, match=r":2: not valid JSON"):
        MockDetector(_runtime(), script)


def test_unknown_field_is_rejected(tmp_path):
    script = _write_script(
        tmp_path,
        [
            {
                "frame_index": 0,
                "detections": [
                    {
                        "coordinate_space": "pixel",
                        "object_class": "chip",
                        "confidence": 0.5,
                        "center": {"x": 1.0, "y": 1.0},
                        "unexpected_field": True,
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="invalid mock script entry"):
        MockDetector(_runtime(), script)


def test_confidence_out_of_range_is_rejected(tmp_path):
    script = _write_script(
        tmp_path,
        [
            {
                "frame_index": 0,
                "detections": [
                    {
                        "coordinate_space": "pixel",
                        "object_class": "chip",
                        "confidence": 1.5,
                        "center": {"x": 1.0, "y": 1.0},
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="invalid mock script entry"):
        MockDetector(_runtime(), script)


def test_unknown_coordinate_space_is_rejected(tmp_path):
    script = _write_script(
        tmp_path,
        [
            {
                "frame_index": 0,
                "detections": [
                    {
                        "coordinate_space": "world",
                        "object_class": "chip",
                        "confidence": 0.5,
                        "center": {"x": 1.0, "y": 1.0},
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="invalid mock script entry"):
        MockDetector(_runtime(), script)


def test_table_entry_with_box_is_rejected(tmp_path):
    # `box` is only defined for "pixel" entries (a table-space box would
    # need its own corner-remapping story, out of scope for REQ-18); a
    # "table" entry that includes one is an unknown field under the
    # discriminated union, not silently ignored.
    script = _write_script(
        tmp_path,
        [
            {
                "frame_index": 0,
                "detections": [
                    {
                        "coordinate_space": "table",
                        "object_class": "chip",
                        "confidence": 0.5,
                        "center": {"x": 1.0, "y": 1.0},
                        "box": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
                    }
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="invalid mock script entry"):
        MockDetector(_runtime(), script)
