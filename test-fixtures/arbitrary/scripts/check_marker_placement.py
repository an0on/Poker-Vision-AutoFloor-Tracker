"""Diagnostic for the REQ-19 ArUco wiring test: which zone did each marker land in?

Running the full pipeline (`poker-vision run`) and checking the JSONL export
is a slow feedback loop for "is this marker actually inside a zone" --
`state`'s hysteresis needs several frames before it emits an event at all,
and a marker that misses every zone produces no output whatsoever, not an
error. This script skips straight to the useful answer: for every marker
found in every photo under `test-fixtures/arbitrary/configs/test_arbitrary.
json`'s `source.path`, print its table-plane coordinate and whichever zone
(if any) it falls inside, using the same `point_in_polygon` check
`assignment` uses.

Usage (from repo root):
    uv run python test-fixtures/arbitrary/scripts/check_marker_placement.py
"""

from __future__ import annotations

import glob

import cv2

from poker_vision.calibration.geometry import TablePoint
from poker_vision.calibration.runtime import CalibrationRuntime, load_calibration_runtime
from poker_vision.calibration.topology import point_in_polygon
from poker_vision.capture.frame import Frame
from poker_vision.config import load_config
from poker_vision.detection.mock_aruco import MockArucoDetector
from poker_vision.detection.models import Detection

CONFIG_PATH = "test-fixtures/arbitrary/configs/test_arbitrary.json"


def describe_zone(point: TablePoint, calibration: CalibrationRuntime) -> str:
    for seat in calibration.seats:
        if point_in_polygon(seat.zones.chip_zone, point):
            return f"{seat.seat_id}.chip_zone"
    for seat in calibration.seats:
        if point_in_polygon(seat.zones.player_area, point):
            return f"{seat.seat_id}.player_area (not chip_zone)"
    if point_in_polygon(calibration.zones.board_zone, point):
        return "board_zone"
    if point_in_polygon(calibration.zones.dealer_area, point):
        return "dealer_area"
    return "NO MATCH -- outside every zone"


def describe_detection(detection: Detection, calibration: CalibrationRuntime) -> str:
    zone = describe_zone(detection.center, calibration)
    return (
        f"  {detection.object_class.value:14s} table=({detection.center.x:7.1f}, "
        f"{detection.center.y:7.1f})  -> {zone}"
    )


def main() -> None:
    config = load_config(CONFIG_PATH)
    calibration = load_calibration_runtime(config.paths.calibration_runtime)
    assert config.aruco is not None, f"{CONFIG_PATH} must configure detector.aruco"
    detector = MockArucoDetector(calibration, config.aruco)

    paths = sorted(
        p
        for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(f"{config.source.path}/{ext}")
    )
    if not paths:
        print(f"no images found in {config.source.path}")
        return

    for index, path in enumerate(paths):
        image = cv2.imread(path)
        if image is None:
            print(f"{path}: unreadable")
            continue
        height, width = image.shape[:2]
        expected = calibration.inference_resolution
        if (width, height) != (expected.width, expected.height):
            print(
                f"{path}: {width}x{height} != required {expected.width}x{expected.height} "
                "-- run test-fixtures/arbitrary/scripts/prepare_test_frames.py first"
            )
            continue
        frame = Frame(image=image, frame_index=index, timestamp=0.0, source_id=path)
        result = detector.detect(frame)
        print(f"{path}: {len(result.detections)} marker(s)")
        if not result.detections:
            print("  (none detected -- check focus/blur/lighting)")
        for detection in result.detections:
            print(describe_detection(detection, calibration))


if __name__ == "__main__":
    main()
