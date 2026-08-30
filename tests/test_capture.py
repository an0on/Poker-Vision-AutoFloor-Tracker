from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from poker_vision.capture import (
    Capture,
    ContinuityCapture,
    Frame,
    ImageDirCapture,
    VideoFileCapture,
    create_capture,
)
from poker_vision.capture.resolution import apply_resolution_cap
from poker_vision.config import Resolution, SourceConfig, SourceType

CAP = Resolution(width=1920, height=1080)


def _write_image(path: Path, width: int, height: int, fill: int) -> None:
    image = np.full((height, width, 3), fill, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _make_image_dir(tmp_path: Path, count: int = 3) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for i in range(count):
        _write_image(image_dir / f"frame_{i:03d}.png", 320, 240, fill=i * 10)
    return image_dir


def _make_video_file(tmp_path: Path, count: int = 5, fps: float = 10.0) -> Path:
    video_path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240)
    )
    for i in range(count):
        writer.write(np.full((240, 320, 3), i * 10, dtype=np.uint8))
    writer.release()
    return video_path


class FakeVideoCapture:
    """Hardware-free stand-in for `cv2.VideoCapture` (REQ-16)."""

    def __init__(self, opened: bool, frames: list[np.ndarray] | None = None) -> None:
        self._opened = opened
        self._frames = frames or []
        self._pos = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._pos >= len(self._frames):
            return False, None
        frame = self._frames[self._pos]
        self._pos += 1
        return True, frame

    def release(self) -> None:
        self.released = True


# --- Frame / Capture shape -------------------------------------------------


def test_frame_has_identical_shape_across_sources(tmp_path):
    image_dir = _make_image_dir(tmp_path, count=1)
    with ImageDirCapture(image_dir, CAP) as capture:
        frame = next(capture)
    assert isinstance(frame, Frame)
    assert isinstance(frame.image, np.ndarray)
    assert frame.frame_index == 0
    assert frame.source_id.startswith("image_dir:")
    assert frame.timestamp is not None


def test_image_dir_capture_is_a_capture(tmp_path):
    image_dir = _make_image_dir(tmp_path, count=1)
    with ImageDirCapture(image_dir, CAP) as capture:
        assert isinstance(capture, Capture)


# --- REQ-14 / AC-9: resolution cap -----------------------------------------


def test_resolution_cap_downscales_preserving_aspect_ratio():
    image = np.zeros((1000, 4000, 3), dtype=np.uint8)  # height, width
    cap = Resolution(width=1920, height=1080)
    scaled = apply_resolution_cap(image, cap)
    height, width = scaled.shape[:2]
    assert width == 1920
    assert height == 480
    assert abs((width / height) - (4000 / 1000)) < 1e-9


def test_resolution_cap_does_not_upscale_smaller_image():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    scaled = apply_resolution_cap(image, CAP)
    assert scaled.shape[:2] == (240, 320)


def test_image_dir_frames_are_capped(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "big.png", 4000, 1000, fill=1)
    small_cap = Resolution(width=1920, height=1080)
    with ImageDirCapture(image_dir, small_cap) as capture:
        frame = next(capture)
    height, width = frame.image.shape[:2]
    assert width == 1920
    assert height == 480


# --- REQ-13, REQ-15, AC-8: image_dir determinism ----------------------------


def test_image_dir_capture_deterministic_sequence_and_indices(tmp_path):
    image_dir = _make_image_dir(tmp_path, count=4)

    def collect() -> list[Frame]:
        with ImageDirCapture(image_dir, CAP) as capture:
            return list(capture)

    first_run = collect()
    second_run = collect()

    assert [f.frame_index for f in first_run] == [0, 1, 2, 3]
    assert [f.frame_index for f in first_run] == [f.frame_index for f in second_run]
    assert [f.timestamp for f in first_run] == [f.timestamp for f in second_run]
    assert [f.source_id for f in first_run] == [f.source_id for f in second_run]
    for a, b in zip(first_run, second_run, strict=True):
        assert np.array_equal(a.image, b.image)


def test_image_dir_capture_timestamps_strictly_increasing(tmp_path):
    image_dir = _make_image_dir(tmp_path, count=3)
    with ImageDirCapture(image_dir, CAP) as capture:
        frames = list(capture)
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)


def test_image_dir_capture_sorted_by_filename_not_creation_order(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    # Write "b" before "a" on disk; sequence must still follow filename order.
    _write_image(image_dir / "b_second.png", 320, 240, fill=1)
    _write_image(image_dir / "a_first.png", 320, 240, fill=2)
    with ImageDirCapture(image_dir, CAP) as capture:
        frames = list(capture)
    assert [f.image[0, 0, 0] for f in frames] == [2, 1]


def test_image_dir_capture_empty_directory_raises(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    with pytest.raises(ValueError, match="no images found"):
        ImageDirCapture(image_dir, CAP)


def test_image_dir_capture_raises_stop_iteration_when_exhausted(tmp_path):
    image_dir = _make_image_dir(tmp_path, count=1)
    with ImageDirCapture(image_dir, CAP) as capture:
        next(capture)
        with pytest.raises(StopIteration):
            next(capture)


# --- REQ-13, REQ-15, AC-8: video_file determinism ---------------------------


def test_video_file_capture_deterministic_sequence_and_indices(tmp_path):
    video_path = _make_video_file(tmp_path, count=5)

    def collect() -> list[Frame]:
        with VideoFileCapture(video_path, CAP) as capture:
            return list(capture)

    first_run = collect()
    second_run = collect()

    assert len(first_run) == 5
    assert [f.frame_index for f in first_run] == [0, 1, 2, 3, 4]
    assert [f.frame_index for f in first_run] == [f.frame_index for f in second_run]
    assert [f.timestamp for f in first_run] == [f.timestamp for f in second_run]


def test_video_file_capture_timestamps_strictly_increasing(tmp_path):
    video_path = _make_video_file(tmp_path, count=4, fps=10.0)
    with VideoFileCapture(video_path, CAP) as capture:
        frames = list(capture)
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)


def test_video_file_capture_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoFileCapture(tmp_path / "does_not_exist.avi", CAP)


def test_video_file_capture_releases_decoder_on_exhaustion(tmp_path):
    video_path = _make_video_file(tmp_path, count=1)
    capture = VideoFileCapture(video_path, CAP)  # no `with`, on purpose
    next(capture)
    with pytest.raises(StopIteration):
        next(capture)
    assert not capture._cap.isOpened()


def test_video_file_capture_frames_are_capped(tmp_path):
    video_path = _make_video_file(tmp_path, count=1)
    small_cap = Resolution(width=160, height=120)
    with VideoFileCapture(video_path, small_cap) as capture:
        frame = next(capture)
    height, width = frame.image.shape[:2]
    assert width <= 160
    assert height <= 120


# --- REQ-13, REQ-16: continuity, hardware-free ------------------------------


def test_continuity_capture_missing_camera_raises_no_fallback():
    def factory(_device_index: int) -> FakeVideoCapture:
        return FakeVideoCapture(opened=False)

    with pytest.raises(RuntimeError, match="not available"):
        ContinuityCapture(0, CAP, capture_factory=factory)


def test_continuity_capture_yields_frames_from_injected_backend():
    frames = [np.full((240, 320, 3), i, dtype=np.uint8) for i in range(3)]
    fake = FakeVideoCapture(opened=True, frames=frames)

    with ContinuityCapture(0, CAP, capture_factory=lambda _idx: fake) as capture:
        # `continuity` is a live source and never raises StopIteration
        # (REQ-16), so pull exactly the fake backend's frame count rather
        # than draining via list().
        collected = [next(capture) for _ in range(len(frames))]

    assert [f.frame_index for f in collected] == [0, 1, 2]
    assert all(f.source_id == "continuity:0" for f in collected)
    assert fake.released


def test_continuity_capture_read_failure_raises_runtime_error():
    fake = FakeVideoCapture(opened=True, frames=[])
    with ContinuityCapture(0, CAP, capture_factory=lambda _idx: fake) as capture:
        with pytest.raises(RuntimeError, match="failed to read frame"):
            next(capture)


def test_continuity_capture_close_releases_underlying_capture():
    fake = FakeVideoCapture(opened=True, frames=[])
    capture = ContinuityCapture(0, CAP, capture_factory=lambda _idx: fake)
    capture.close()
    assert fake.released


# --- create_capture factory --------------------------------------------------


def test_create_capture_dispatches_image_dir(tmp_path):
    image_dir = _make_image_dir(tmp_path, count=1)
    source = SourceConfig(type=SourceType.IMAGE_DIR, path=image_dir)
    capture = create_capture(source)
    try:
        assert isinstance(capture, ImageDirCapture)
    finally:
        capture.close()


def test_create_capture_dispatches_video_file(tmp_path):
    video_path = _make_video_file(tmp_path, count=1)
    source = SourceConfig(type=SourceType.VIDEO_FILE, path=video_path)
    capture = create_capture(source)
    try:
        assert isinstance(capture, VideoFileCapture)
    finally:
        capture.close()


def test_create_capture_dispatches_continuity_no_fallback_on_missing_device():
    # No real camera required: an implausible device index fails to open on
    # any platform/backend, exercising exactly the REQ-16 error path.
    source = SourceConfig(type=SourceType.CONTINUITY, device_index=99)
    with pytest.raises(RuntimeError, match="not available"):
        create_capture(source)
