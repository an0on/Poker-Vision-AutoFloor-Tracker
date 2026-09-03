"""REQ-10b: `calib learn-table` (AC-6b)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from poker_vision.calibration.camera import CameraIntrinsics, DistortionCoefficients
from poker_vision.calibration.geometry import (
    PixelPoint,
    TableDimensions,
    TablePoint,
    TablePolygon,
    TableUnit,
)
from poker_vision.calibration.homography import HomographyMatrix
from poker_vision.calibration.learn_table import (
    LearnTableConfig,
    LearnTableError,
    _filter_reliable_matches,
    learn_table_calibration,
)
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.calibration.zones import CalibrationSeat, GlobalZones, SeatZones
from poker_vision.detection.geometry import apply_homography_to_point

WIDTH, HEIGHT = 1200, 900
_IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

# `dealer_area`/`board_zone` sit in the middle of the table -- the synthetic
# reference photo's "printed" texture (a field of random circles, standing
# in for a real table's card outline/branding) is drawn to line up with
# them, same as a real table's is drawn to line up with its own print.
_BOARD_ZONE = TablePolygon(
    points=[
        TablePoint(x=550, y=400),
        TablePoint(x=650, y=400),
        TablePoint(x=650, y=500),
        TablePoint(x=550, y=500),
    ]
)
_DEALER_AREA = TablePolygon(
    points=[
        TablePoint(x=300, y=200),
        TablePoint(x=900, y=200),
        TablePoint(x=900, y=700),
        TablePoint(x=300, y=700),
    ]
)
# Off in a corner, far from the center strip -- doesn't need to resemble a
# real seat, just needs to be a REQ-11-valid zone so `CalibrationRuntime`
# validates.
_SEAT = CalibrationSeat(
    seat_id="seat_1",
    zones=SeatZones(
        player_area=TablePolygon(
            points=[
                TablePoint(x=0, y=0),
                TablePoint(x=200, y=0),
                TablePoint(x=200, y=150),
                TablePoint(x=0, y=150),
            ]
        ),
        chip_zone=TablePolygon(
            points=[
                TablePoint(x=20, y=20),
                TablePoint(x=150, y=20),
                TablePoint(x=150, y=120),
                TablePoint(x=20, y=120),
            ]
        ),
    ),
)


def _reference_runtime() -> CalibrationRuntime:
    # Identity homography + zero distortion, same trick as the fixtures in
    # test_calibration_compile.py / test_detection_mock_coco.py: image
    # pixel == table coordinate, so recovered table coordinates can be
    # checked directly against known reference-image pixel positions.
    return CalibrationRuntime(
        schema_version="1.1",
        table_id="reference_table",
        based_on="x",
        inference_resolution={"width": WIDTH, "height": HEIGHT},
        camera=CameraIntrinsics(fx=1400.0, fy=1400.0, cx=WIDTH / 2, cy=HEIGHT / 2),
        distortion=DistortionCoefficients(),
        homography=HomographyMatrix(forward=_IDENTITY, inverse=_IDENTITY),
        table=TableDimensions(width=WIDTH, height=HEIGHT, unit=TableUnit.MM),
        seats=[_SEAT],
        zones=GlobalZones(board_zone=_BOARD_ZONE, dealer_area=_DEALER_AREA),
        card_dealer_seat_id="seat_1",
    )


def _synthetic_reference_image() -> np.ndarray:
    """A flat "felt" background with a field of random circles standing in
    for a real table's printed card outline/branding, positioned so it
    covers `_DEALER_AREA` (with room to spare) -- rich enough in corners
    for ORB to find plenty of ORB keypoints, the same way real print does.
    """
    rng = np.random.default_rng(42)
    image = np.full((HEIGHT, WIDTH), 100, dtype=np.uint8)
    for _ in range(250):
        cx = int(rng.integers(260, 940))
        cy = int(rng.integers(160, 740))
        radius = int(rng.integers(3, 14))
        color = int(rng.integers(20, 230))
        cv2.circle(image, (cx, cy), radius, color, -1)
    return image


def _warp_matrix() -> np.ndarray:
    """A modest rotation + scale + translation -- simulates a second photo
    of the same table taken from a slightly different camera position, not
    a pathological transform.
    """
    affine = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), angle=7.0, scale=0.8)
    affine[0, 2] += 40
    affine[1, 2] += -20
    return np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float64)


def _apply_matrix(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    vector = matrix @ np.array([x, y, 1.0])
    return float(vector[0] / vector[2]), float(vector[1] / vector[2])


@pytest.fixture
def reference_photo(tmp_path):
    image = _synthetic_reference_image()
    path = tmp_path / "reference.png"
    cv2.imwrite(str(path), image)
    return path, image


@pytest.fixture
def live_photo(tmp_path, reference_photo):
    _, reference_image = reference_photo
    warp = _warp_matrix()
    live_image = cv2.warpPerspective(reference_image, warp, (WIDTH, HEIGHT), borderValue=100)
    path = tmp_path / "live.png"
    cv2.imwrite(str(path), live_image)
    return path, warp


# --- AC-6b: recovered table coordinates match within tolerance --------------


def test_learn_table_recovers_known_points_within_tolerance(reference_photo, live_photo):
    reference_path, _ = reference_photo
    live_path, warp = live_photo
    reference = _reference_runtime()

    runtime = learn_table_calibration(
        reference,
        reference_image_path=reference_path,
        live_image_path=live_path,
        based_on="test",
    )

    # AC-6b: tolerance <= 1% of table width.
    tolerance = 0.01 * reference.table.width
    known_reference_points = [(600, 450), (500, 350), (700, 550), (400, 300)]
    for rx, ry in known_reference_points:
        live_x, live_y = _apply_matrix(warp, rx, ry)
        recovered = apply_homography_to_point(
            PixelPoint(x=live_x, y=live_y), runtime.homography, runtime.camera, runtime.distortion
        )
        assert recovered.x == pytest.approx(rx, abs=tolerance)
        assert recovered.y == pytest.approx(ry, abs=tolerance)


def test_learn_table_output_is_req11_valid(reference_photo, live_photo):
    reference_path, _ = reference_photo
    live_path, _ = live_photo
    runtime = learn_table_calibration(
        _reference_runtime(),
        reference_image_path=reference_path,
        live_image_path=live_path,
        based_on="test",
    )
    # Construction already ran full CalibrationRuntime/REQ-11 validation;
    # re-validating the round-tripped JSON pins that nothing about the
    # composed homography only "happens" to validate on the live object.
    reloaded = CalibrationRuntime.model_validate_json(runtime.model_dump_json())
    assert reloaded == runtime


def test_learn_table_carries_zones_seats_and_camera_unchanged(reference_photo, live_photo):
    reference_path, _ = reference_photo
    live_path, _ = live_photo
    reference = _reference_runtime()
    runtime = learn_table_calibration(
        reference,
        reference_image_path=reference_path,
        live_image_path=live_path,
        based_on="test",
    )
    assert runtime.zones == reference.zones
    assert runtime.seats == reference.seats
    assert runtime.card_dealer_seat_id == reference.card_dealer_seat_id
    assert runtime.camera == reference.camera
    assert runtime.distortion == reference.distortion
    assert runtime.table == reference.table
    assert runtime.table_id == reference.table_id


def test_learn_table_accepts_table_id_override(reference_photo, live_photo):
    reference_path, _ = reference_photo
    live_path, _ = live_photo
    runtime = learn_table_calibration(
        _reference_runtime(),
        reference_image_path=reference_path,
        live_image_path=live_path,
        based_on="test",
        table_id="new_physical_table",
    )
    assert runtime.table_id == "new_physical_table"


# --- REQ-10b step 3 / AC-6b: clear abort on unreliable input -----------------


def test_learn_table_rejects_missing_reference_image(tmp_path, live_photo):
    live_path, _ = live_photo
    with pytest.raises(LearnTableError):
        learn_table_calibration(
            _reference_runtime(),
            reference_image_path=tmp_path / "does_not_exist.png",
            live_image_path=live_path,
            based_on="test",
        )


def test_learn_table_rejects_wrong_resolution_live_image(tmp_path, reference_photo):
    reference_path, _ = reference_photo
    wrong_size = np.full((HEIGHT // 2, WIDTH // 2), 100, dtype=np.uint8)
    live_path = tmp_path / "wrong_size.png"
    cv2.imwrite(str(live_path), wrong_size)
    with pytest.raises(LearnTableError, match="resolution"):
        learn_table_calibration(
            _reference_runtime(),
            reference_image_path=reference_path,
            live_image_path=live_path,
            based_on="test",
        )


def test_learn_table_rejects_blank_live_photo_too_few_matches(tmp_path, reference_photo):
    # A blank/featureless live photo (e.g. camera pointed at nothing, or a
    # wildly different scene) has no ORB keypoints at all, so it can't
    # produce any feature matches against the reference's textured center
    # strip.
    reference_path, _ = reference_photo
    blank = np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)
    live_path = tmp_path / "blank.png"
    cv2.imwrite(str(live_path), blank)
    with pytest.raises(LearnTableError, match="no ORB features detected"):
        learn_table_calibration(
            _reference_runtime(),
            reference_image_path=reference_path,
            live_image_path=live_path,
            based_on="test",
        )


def test_learn_table_rejects_unrelated_live_photo_low_inlier_ratio(tmp_path, reference_photo):
    # A different random texture has plenty of ORB features of its own
    # (enough to pass the raw match-count filter via incidental descriptor
    # similarity) but no real geometric relationship to the reference --
    # RANSAC should find too few inliers to trust the result.
    reference_path, _ = reference_photo
    rng = np.random.default_rng(7)
    unrelated = np.full((HEIGHT, WIDTH), 100, dtype=np.uint8)
    for _ in range(250):
        cx = int(rng.integers(0, WIDTH))
        cy = int(rng.integers(0, HEIGHT))
        radius = int(rng.integers(3, 14))
        color = int(rng.integers(20, 230))
        cv2.circle(unrelated, (cx, cy), radius, color, -1)
    live_path = tmp_path / "unrelated.png"
    cv2.imwrite(str(live_path), unrelated)
    with pytest.raises(LearnTableError):
        learn_table_calibration(
            _reference_runtime(),
            reference_image_path=reference_path,
            live_image_path=live_path,
            based_on="test",
            config=LearnTableConfig(min_match_count=4),
        )


# --- _filter_reliable_matches: one-to-one deduplication ----------------------


def _dmatch(query_idx: int, train_idx: int, distance: float) -> cv2.DMatch:
    return cv2.DMatch(query_idx, train_idx, distance)


def test_filter_reliable_matches_keeps_best_per_reference_keypoint():
    # Live keypoints 0 and 1 both pass the ratio test against reference
    # keypoint 5 (a repeated pattern in the live photo, e.g. AC-6b's
    # concern about symmetric print/branding) -- only the closer
    # (lower-distance) one, live keypoint 0, should survive.
    raw_matches = [
        [_dmatch(0, 5, 10.0), _dmatch(0, 6, 40.0)],
        [_dmatch(1, 5, 20.0), _dmatch(1, 7, 40.0)],
        [_dmatch(2, 8, 5.0), _dmatch(2, 9, 40.0)],
    ]
    good = _filter_reliable_matches(raw_matches, min_match_count=1)
    train_indices = [m.trainIdx for m in good]
    assert len(train_indices) == len(set(train_indices))
    assert any(m.queryIdx == 0 and m.trainIdx == 5 for m in good)
    assert not any(m.queryIdx == 1 for m in good)


def test_filter_reliable_matches_rejects_ambiguous_ratio_test_matches():
    # Best and second-best reference candidates are nearly equidistant --
    # Lowe's ratio test can't tell them apart, so neither counts.
    raw_matches = [[_dmatch(0, 1, 10.0), _dmatch(0, 2, 10.5)]]
    with pytest.raises(LearnTableError, match="too few"):
        _filter_reliable_matches(raw_matches, min_match_count=1)


def test_filter_reliable_matches_raises_below_min_match_count():
    raw_matches = [[_dmatch(0, 1, 1.0), _dmatch(0, 2, 100.0)]]
    with pytest.raises(LearnTableError, match=r"need >= 5, got 1"):
        _filter_reliable_matches(raw_matches, min_match_count=5)


# --- LearnTableConfig validation ---------------------------------------------


def test_learn_table_config_rejects_too_few_min_match_count():
    with pytest.raises(LearnTableError, match="min_match_count"):
        LearnTableConfig(min_match_count=3)


@pytest.mark.parametrize("bad_ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_learn_table_config_rejects_invalid_min_inlier_ratio(bad_ratio):
    with pytest.raises(LearnTableError, match="min_inlier_ratio"):
        LearnTableConfig(min_inlier_ratio=bad_ratio)


@pytest.mark.parametrize("bad_threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_learn_table_config_rejects_invalid_ransac_reproj_threshold(bad_threshold):
    with pytest.raises(LearnTableError, match="ransac_reproj_threshold"):
        LearnTableConfig(ransac_reproj_threshold=bad_threshold)


@pytest.mark.parametrize("bad_margin", [-0.5, float("nan"), float("inf")])
def test_learn_table_config_rejects_invalid_center_strip_margin_ratio(bad_margin):
    with pytest.raises(LearnTableError, match="center_strip_margin_ratio"):
        LearnTableConfig(center_strip_margin_ratio=bad_margin)


def test_learn_table_based_on_is_carried_through(reference_photo, live_photo):
    reference_path, _ = reference_photo
    live_path, _ = live_photo
    runtime = learn_table_calibration(
        _reference_runtime(),
        reference_image_path=reference_path,
        live_image_path=live_path,
        based_on="some/traceable/identifier",
    )
    assert runtime.based_on == "some/traceable/identifier"
