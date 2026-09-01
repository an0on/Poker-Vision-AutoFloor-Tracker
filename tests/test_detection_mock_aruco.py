"""REQ-19: `mock` detector, Modus B (ArUco markers), AC-11 slice."""

from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from poker_vision.calibration.geometry import PixelPoint
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.config import ArucoDetectionConfig, ArucoDictionary
from poker_vision.detection.geometry import apply_homography_to_point
from poker_vision.detection.mock_aruco import MockArucoDetector
from poker_vision.detection.models import DetectionClass

# Identity-ish homography with real distortion, so the pixel -> table
# transform is actually exercised (mirrors test_detection_mock.py).
FORWARD = [[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]]
INVERSE = [[0.5, 0.0, -5.0], [0.0, 1.0 / 3.0, -20.0 / 3.0], [0.0, 0.0, 1.0]]
CAMERA = {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0}
DISTORTION = {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0}

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

IMAGE_SIZE = 1920, 1080  # width, height


def _runtime() -> CalibrationRuntime:
    payload = {
        "schema_version": "1.1",
        "table_id": "test_table",
        "based_on": "calibration/instance.json",
        "inference_resolution": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]},
        "camera": CAMERA,
        "distortion": DISTORTION,
        "homography": {"forward": FORWARD, "inverse": INVERSE},
        "table": {"width": 1200.0, "height": 900.0, "unit": "mm"},
        "seats": VALID_SEATS,
        "zones": VALID_ZONES,
        "card_dealer_seat_id": "seat_1",
    }
    return CalibrationRuntime.model_validate(payload)


def _frame(image: np.ndarray, frame_index: int = 0) -> Frame:
    return Frame(
        image=image,
        timestamp=datetime.now(UTC),
        frame_index=frame_index,
        source_id="test",
    )


def _blank_image() -> np.ndarray:
    return np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 255, dtype=np.uint8)


def _draw_marker(
    image: np.ndarray,
    dictionary_id: int,
    marker_id: int,
    top_left: tuple[int, int],
    size: int = 200,
) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    x, y = top_left
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    image[y : y + size, x : x + size] = marker_bgr


def _config(marker_class_map: dict[int, str]) -> ArucoDetectionConfig:
    return ArucoDetectionConfig(
        dictionary=ArucoDictionary.DICT_4X4_50, marker_class_map=marker_class_map
    )


# --- basic marker -> detection mapping ---------------------------------------


def test_mapped_marker_yields_detection_with_configured_class():
    image = _blank_image()
    _draw_marker(image, cv2.aruco.DICT_4X4_50, marker_id=7, top_left=(100, 100), size=200)
    detector = MockArucoDetector(_runtime(), _config({7: "chip"}))

    result = detector.detect(_frame(image))

    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.object_class is DetectionClass.CHIP
    assert detection.confidence == pytest.approx(1.0)
    assert detection.box is None


def test_marker_center_matches_mean_of_corners_transformed_to_table():
    image = _blank_image()
    # Marker spans pixel columns/rows [100, 300), so its true centre is
    # (199.5, 199.5) -- verified directly against cv2.aruco in isolation
    # before writing this test.
    _draw_marker(image, cv2.aruco.DICT_4X4_50, marker_id=3, top_left=(100, 100), size=200)
    calibration = _runtime()
    detector = MockArucoDetector(calibration, _config({3: "dealer_button"}))

    result = detector.detect(_frame(image))

    expected = apply_homography_to_point(
        PixelPoint(x=199.5, y=199.5),
        calibration.homography,
        calibration.camera,
        calibration.distortion,
    )
    detection = result.detections[0]
    assert detection.center.x == pytest.approx(expected.x, abs=1e-2)
    assert detection.center.y == pytest.approx(expected.y, abs=1e-2)


def test_unmapped_marker_id_is_skipped():
    image = _blank_image()
    _draw_marker(image, cv2.aruco.DICT_4X4_50, marker_id=42, top_left=(100, 100), size=200)
    detector = MockArucoDetector(_runtime(), _config({0: "chip"}))

    result = detector.detect(_frame(image))

    assert result.detections == []


def test_no_markers_in_frame_yields_no_detections():
    detector = MockArucoDetector(_runtime(), _config({0: "chip"}))

    result = detector.detect(_frame(_blank_image()))

    assert result.detections == []


def test_multiple_markers_map_to_their_configured_classes():
    image = _blank_image()
    _draw_marker(image, cv2.aruco.DICT_4X4_50, marker_id=1, top_left=(50, 50), size=150)
    _draw_marker(image, cv2.aruco.DICT_4X4_50, marker_id=2, top_left=(800, 600), size=150)
    detector = MockArucoDetector(_runtime(), _config({1: "chip", 2: "card"}))

    result = detector.detect(_frame(image))

    classes = {detection.object_class for detection in result.detections}
    assert classes == {DetectionClass.CHIP, DetectionClass.CARD}
    assert len(result.detections) == 2


def test_frame_index_is_preserved():
    detector = MockArucoDetector(_runtime(), _config({0: "chip"}))

    result = detector.detect(_frame(_blank_image(), frame_index=12))

    assert result.frame_index == 12


# --- AC-11: agreement with a pixel-space entry for the same physical point --


def test_aruco_center_agrees_with_equivalent_pixel_detection():
    # Same physical marker centre (199.5, 199.5), once produced by the ArUco
    # detector and once fed directly through the shared pixel -> table
    # transform (what a Modus A "pixel" script entry would use) -- both must
    # land within AC-11's tolerance (<= 1% of table width) of each other.
    image = _blank_image()
    _draw_marker(image, cv2.aruco.DICT_4X4_50, marker_id=5, top_left=(100, 100), size=200)
    calibration = _runtime()
    detector = MockArucoDetector(calibration, _config({5: "chip"}))

    aruco_result = detector.detect(_frame(image)).detections[0]
    direct_result = apply_homography_to_point(
        PixelPoint(x=199.5, y=199.5),
        calibration.homography,
        calibration.camera,
        calibration.distortion,
    )

    tolerance = 0.01 * calibration.table.width
    assert aruco_result.center.x == pytest.approx(direct_result.x, abs=tolerance)
    assert aruco_result.center.y == pytest.approx(direct_result.y, abs=tolerance)


# --- config validation --------------------------------------------------------


def test_empty_marker_class_map_is_rejected():
    with pytest.raises(ValidationError, match="marker_class_map"):
        ArucoDetectionConfig(marker_class_map={})


def test_unknown_object_class_in_marker_class_map_is_rejected():
    with pytest.raises(ValidationError):
        ArucoDetectionConfig(marker_class_map={0: "not_a_real_class"})


def test_default_dictionary_is_4x4_50():
    config = ArucoDetectionConfig(marker_class_map={0: "chip"})
    assert config.dictionary is ArucoDictionary.DICT_4X4_50


def test_non_default_dictionary_is_used_for_detection():
    image = _blank_image()
    _draw_marker(image, cv2.aruco.DICT_6X6_250, marker_id=9, top_left=(100, 100), size=200)
    config = ArucoDetectionConfig(
        dictionary=ArucoDictionary.DICT_6X6_250, marker_class_map={9: "card"}
    )
    detector = MockArucoDetector(_runtime(), config)

    result = detector.detect(_frame(image))

    assert len(result.detections) == 1
    assert result.detections[0].object_class is DetectionClass.CARD


def test_wrong_dictionary_does_not_detect_marker():
    # A DICT_6X6_250 marker isn't a valid symbol in DICT_4X4_50, so the
    # detector configured for the wrong family must find nothing here --
    # not misread it as some other marker ID.
    image = _blank_image()
    _draw_marker(image, cv2.aruco.DICT_6X6_250, marker_id=9, top_left=(100, 100), size=200)
    detector = MockArucoDetector(_runtime(), _config({9: "card"}))

    result = detector.detect(_frame(image))

    assert result.detections == []
