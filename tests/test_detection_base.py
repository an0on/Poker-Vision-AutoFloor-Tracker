"""REQ-17: Detector interface, pixel -> table transform, AC-10."""

from __future__ import annotations

from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import PixelPoint
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.detection.base import Detector, RawDetection
from poker_vision.detection.geometry import (
    apply_homography_to_point,
    box_center,
    transform_box_to_table,
)
from poker_vision.detection.models import DetectionClass

# Zero distortion: apply_homography_to_point/transform_box_to_table always
# undistort first, and these geometry-only tests want to isolate the
# homography math, so distortion is neutralised rather than exercised here
# (the `Detector.detect()` tests below cover the combined pipeline).
ZERO_DISTORTION = DistortionCoefficients()
NEUTRAL_CAMERA = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0)

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

# Scale x by 2, y by 3, then translate by (10, 20): table = (2x + 10, 3y + 20).
# Deliberately not the identity matrix, so a test can't pass just because the
# transform happens to be a no-op.
SCALE_TRANSLATE_FORWARD = [[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]]
SCALE_TRANSLATE_INVERSE = [[0.5, 0.0, -5.0], [0.0, 1.0 / 3.0, -20.0 / 3.0], [0.0, 0.0, 1.0]]

# Rotate 90 degrees about the origin: (x, y) -> (-y, x). Used to check that
# the box transform takes the bounding box of all four transformed corners,
# not a literal corner-to-corner remap (a homography need not stay axis-aligned).
ROTATE_90_FORWARD = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
ROTATE_90_INVERSE = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def _runtime(forward: list[list[float]], inverse: list[list[float]]) -> CalibrationRuntime:
    payload = {
        "schema_version": "1.0",
        "table_id": "test_table",
        "based_on": "calibration/instance.json",
        # Matches _frame()'s 10x10 image, so the resolution guard in
        # Detector.detect() doesn't reject these fixtures.
        "inference_resolution": {"width": 10, "height": 10},
        "camera": {"fx": 1400.0, "fy": 1400.0, "cx": 960.0, "cy": 540.0},
        "distortion": {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        "homography": {"forward": forward, "inverse": inverse},
        "table": {"width": 1200.0, "height": 900.0, "unit": "mm"},
        "seats": VALID_SEATS,
        "zones": VALID_ZONES,
    }
    return CalibrationRuntime.model_validate(payload)


def _frame(frame_index: int = 0) -> Frame:
    return Frame(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        timestamp=datetime.now(UTC),
        frame_index=frame_index,
        source_id="test",
    )


class _StubDetector(Detector):
    """Minimal `Detector` used only to exercise the base class's transform."""

    def __init__(self, calibration: CalibrationRuntime, raw: list[RawDetection]) -> None:
        super().__init__(calibration)
        self._raw = raw

    def _detect_raw(self, frame: Frame) -> list[RawDetection]:
        return self._raw


# --- geometry.box_center: Phase 0's verified method -------------------------


def test_box_center_matches_phase0_method():
    assert box_center((0.0, 0.0, 10.0, 20.0)) == PixelPoint(x=5.0, y=10.0)


def test_box_center_matches_phase0_reference_values():
    # Phase 0's own reported centre for the dealer-button placeholder
    # (docs/phase0 run on Test1.jpeg), reconstructed from its box.
    center = box_center((2094.0, 2814.0, 2414.54, 3015.22))
    assert center.x == pytest.approx(2254.27, abs=1.0)
    assert center.y == pytest.approx(2914.61, abs=1.0)


# --- geometry.apply_homography_to_point / transform_box_to_table -----------


def test_apply_homography_to_point_scale_translate():
    homography = HomographyMatrix(forward=SCALE_TRANSLATE_FORWARD, inverse=SCALE_TRANSLATE_INVERSE)
    result = apply_homography_to_point(
        PixelPoint(x=10.0, y=10.0), homography, NEUTRAL_CAMERA, ZERO_DISTORTION
    )
    assert result.x == pytest.approx(30.0)
    assert result.y == pytest.approx(50.0)


def test_transform_box_to_table_scale_translate():
    homography = HomographyMatrix(forward=SCALE_TRANSLATE_FORWARD, inverse=SCALE_TRANSLATE_INVERSE)
    box = transform_box_to_table(
        (0.0, 0.0, 10.0, 10.0), homography, NEUTRAL_CAMERA, ZERO_DISTORTION
    )
    assert box.min.x == pytest.approx(10.0)
    assert box.min.y == pytest.approx(20.0)
    assert box.max.x == pytest.approx(30.0)
    assert box.max.y == pytest.approx(50.0)


def test_transform_box_to_table_handles_rotation():
    # A homography need not preserve axis alignment; the result must be the
    # bounding box of all four transformed corners, not a corner remap.
    homography = HomographyMatrix(forward=ROTATE_90_FORWARD, inverse=ROTATE_90_INVERSE)
    box = transform_box_to_table(
        (0.0, 0.0, 10.0, 20.0), homography, NEUTRAL_CAMERA, ZERO_DISTORTION
    )
    assert box.min.x == pytest.approx(-20.0)
    assert box.min.y == pytest.approx(0.0)
    assert box.max.x == pytest.approx(0.0)
    assert box.max.y == pytest.approx(10.0)


def test_transform_box_to_table_bounds_curved_edges_from_distortion():
    # Undistortion is nonlinear, so under real distortion a box's straight
    # edges become curves and the true extremum can land strictly between
    # two corners. A corners-only implementation misses that; sampling the
    # edges (the fix) must not.
    camera = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=0.0, cy=0.0)
    distortion = DistortionCoefficients(k1=-0.5)
    identity = HomographyMatrix(
        forward=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        inverse=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    box = (-400.0, 90.0, 400.0, 110.0)

    result = transform_box_to_table(box, identity, camera, distortion)

    # What a corners-only implementation would have produced, computed
    # independently here (not by reaching into geometry.py's internals).
    corners = np.array(
        [[[-400.0, 90.0]], [[400.0, 90.0]], [[400.0, 110.0]], [[-400.0, 110.0]]],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [[camera.fx, 0.0, camera.cx], [0.0, camera.fy, camera.cy], [0.0, 0.0, 1.0]]
    )
    dist_coeffs = np.array(
        [distortion.k1, distortion.k2, distortion.p1, distortion.p2, distortion.k3]
    )
    undistorted_corners = cv2.undistortPoints(corners, camera_matrix, dist_coeffs, P=camera_matrix)
    corner_only_min_y = float(undistorted_corners[:, 0, 1].min())

    # The sampled implementation finds a smaller (more negative-ward) y
    # along an edge than any corner reaches -- exactly what corners-only
    # sampling would miss.
    assert result.min.y < corner_only_min_y - 5.0


def test_apply_homography_to_point_rejects_horizon_point():
    # A real, invertible homography whose last row is [1, 0, 0]: at (0, 0)
    # the homogeneous w = 1*0 + 0*0 + 0 = 0, i.e. this specific point (not
    # the whole matrix) maps to the horizon.
    forward = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    inverse = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]]
    homography = HomographyMatrix(forward=forward, inverse=inverse)
    with pytest.raises(ValueError, match="horizon"):
        apply_homography_to_point(
            PixelPoint(x=0.0, y=0.0), homography, NEUTRAL_CAMERA, ZERO_DISTORTION
        )


def test_apply_homography_to_point_accepts_tiny_but_valid_scale():
    # HomographyMatrix only requires forward @ inverse == identity, which a
    # uniformly-rescaled matrix pair still satisfies -- scaling forward by s
    # and inverse by 1/s leaves their product unchanged. A pre-fix absolute
    # threshold on the raw (unnormalised) w would reject this valid matrix
    # for every ordinary point, since w is tiny for all of them here.
    # Small enough that the pre-fix absolute threshold (1e-9 on the raw,
    # unnormalised w) would have rejected this point (raw w = 1e-10 here);
    # the fix's Frobenius-normalised w is ~0.044, comfortably above it.
    scale = 1e-10
    forward = [
        [2.0 * scale, 0.0, 10.0 * scale],
        [0.0, 3.0 * scale, 20.0 * scale],
        [0.0, 0.0, scale],
    ]
    inverse = [
        [0.5 / scale, 0.0, -5.0 / scale],
        [0.0, (1.0 / 3.0) / scale, (-20.0 / 3.0) / scale],
        [0.0, 0.0, 1.0 / scale],
    ]
    homography = HomographyMatrix(forward=forward, inverse=inverse)

    result = apply_homography_to_point(
        PixelPoint(x=10.0, y=10.0), homography, NEUTRAL_CAMERA, ZERO_DISTORTION
    )

    assert result.x == pytest.approx(30.0)
    assert result.y == pytest.approx(50.0)


def test_apply_homography_to_point_undistorts_before_transform():
    # A camera with real distortion: undistorting (cx, cy) is a no-op
    # (the distortion model is centred there), so the principal point must
    # map through the identity homography unchanged. This fails if
    # apply_homography_to_point skips undistortion (P1 fix).
    camera = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=50.0, cy=40.0)
    distortion = DistortionCoefficients(k1=0.2, k2=0.05)
    identity = HomographyMatrix(
        forward=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        inverse=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    result = apply_homography_to_point(PixelPoint(x=50.0, y=40.0), identity, camera, distortion)
    assert result.x == pytest.approx(50.0, abs=1e-6)
    assert result.y == pytest.approx(40.0, abs=1e-6)


# --- Detector: interface + AC-10 transform boundary -------------------------


def test_detector_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Detector(_runtime(SCALE_TRANSLATE_FORWARD, SCALE_TRANSLATE_INVERSE))  # type: ignore[abstract]


def test_detect_transforms_center_and_box_into_table_coordinates():
    calibration = _runtime(SCALE_TRANSLATE_FORWARD, SCALE_TRANSLATE_INVERSE)
    raw = RawDetection(
        object_class=DetectionClass.CHIP,
        confidence=0.9,
        # Deliberately not box_center(box) = (5, 5): a box's centre must
        # always come from Phase 0's method, so detect() must ignore this
        # value entirely (see test below for the dedicated check).
        center=PixelPoint(x=10.0, y=10.0),
        box=(0.0, 0.0, 10.0, 10.0),
    )
    detector = _StubDetector(calibration, [raw])

    result = detector.detect(_frame(frame_index=7))

    assert result.schema_version == "1.0"
    assert result.frame_index == 7
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.object_class is DetectionClass.CHIP
    assert detection.confidence == 0.9
    # box_center((0, 0, 10, 10)) = (5, 5) -> (2*5+10, 3*5+20) = (20, 35),
    # not the (30, 50) raw.center alone would have produced.
    assert detection.center.x == pytest.approx(20.0)
    assert detection.center.y == pytest.approx(35.0)
    assert detection.box is not None
    assert detection.box.min.x == pytest.approx(10.0)
    assert detection.box.min.y == pytest.approx(20.0)
    assert detection.box.max.x == pytest.approx(30.0)
    assert detection.box.max.y == pytest.approx(50.0)


def test_detect_ignores_raw_center_when_box_is_present():
    # Codex finding: detect() must not trust a subclass-supplied centre that
    # disagrees with Phase 0's box_center method whenever a box exists -
    # every box-based detector has to agree on the same centre method
    # (REQ-17), not just by convention.
    calibration = _runtime(SCALE_TRANSLATE_FORWARD, SCALE_TRANSLATE_INVERSE)
    box = (0.0, 0.0, 10.0, 10.0)  # box_center -> (5, 5)
    wrong_center = RawDetection(
        object_class=DetectionClass.CHIP,
        confidence=0.9,
        center=PixelPoint(x=9999.0, y=9999.0),  # nonsense, must be ignored
        box=box,
    )
    correct_center = RawDetection(
        object_class=DetectionClass.CHIP,
        confidence=0.9,
        center=PixelPoint(x=5.0, y=5.0),  # == box_center(box)
        box=box,
    )

    detector = _StubDetector(calibration, [wrong_center])
    result_with_wrong_center = detector.detect(_frame())
    detector = _StubDetector(calibration, [correct_center])
    result_with_correct_center = detector.detect(_frame())

    detection_a = result_with_wrong_center.detections[0]
    detection_b = result_with_correct_center.detections[0]
    assert detection_a.center.x == pytest.approx(detection_b.center.x)
    assert detection_a.center.y == pytest.approx(detection_b.center.y)


def test_detect_without_raw_box_leaves_detection_box_none():
    calibration = _runtime(SCALE_TRANSLATE_FORWARD, SCALE_TRANSLATE_INVERSE)
    raw = RawDetection(
        object_class=DetectionClass.DEALER_BUTTON,
        confidence=0.5,
        center=PixelPoint(x=0.0, y=0.0),
    )
    detector = _StubDetector(calibration, [raw])

    result = detector.detect(_frame())

    assert result.detections[0].box is None


def test_detect_empty_raw_detections_yields_empty_frame():
    calibration = _runtime(SCALE_TRANSLATE_FORWARD, SCALE_TRANSLATE_INVERSE)
    detector = _StubDetector(calibration, [])

    result = detector.detect(_frame(frame_index=3))

    assert result.frame_index == 3
    assert result.detections == []


def test_detect_rejects_frame_resolution_mismatch():
    # _runtime()'s calibration is authored against a 10x10 inference
    # resolution (matching _frame()); a frame of any other size means the
    # homography/camera intrinsics no longer apply to its pixel grid (P2
    # fix) and must be rejected rather than silently mis-transformed.
    calibration = _runtime(SCALE_TRANSLATE_FORWARD, SCALE_TRANSLATE_INVERSE)
    detector = _StubDetector(calibration, [])
    mismatched_frame = Frame(
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        timestamp=datetime.now(UTC),
        frame_index=0,
        source_id="test",
    )

    with pytest.raises(ValueError, match="resolution"):
        detector.detect(mismatched_frame)
