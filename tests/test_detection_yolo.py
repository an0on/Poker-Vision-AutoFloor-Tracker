"""REQ-22: `yolo` detector is a registered `Detector` implementation with no model."""

from __future__ import annotations

import pytest

from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.detection.base import Detector
from poker_vision.detection.yolo import YoloDetector

CAMERA = {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0}
DISTORTION = {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}
FORWARD = [[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]]
INVERSE = [[0.5, 0.0, -5.0], [0.0, 1.0 / 3.0, -20.0 / 3.0], [0.0, 0.0, 1.0]]


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
        "seats": [
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
        ],
        "zones": {
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
        },
        "card_dealer_seat_id": "seat_1",
    }
    return CalibrationRuntime.model_validate(payload)


# REQ-22: registered on the Detector interface, even though it can't be built in v0.1.
def test_yolo_detector_is_a_detector_subclass():
    assert issubclass(YoloDetector, Detector)


# AC-13: constructing it directly fails clearly, pointing at v0.2 (defense in
# depth alongside Config._reject_yolo_detector, which is the path v0.1 actually uses).
def test_yolo_detector_construction_raises_with_v02_hint():
    with pytest.raises(NotImplementedError, match="not available in v0.1") as exc_info:
        YoloDetector(_runtime())
    assert "v0.2" in str(exc_info.value)
