"""Capture stage: turns a configured source into a `Frame` stream (REQ-13).

`continuity`, `video_file` and `image_dir` all implement the same `Capture`
interface and produce the identical `Frame` shape (image, timestamp,
running frame index, source id) regardless of source.
"""

from __future__ import annotations

from poker_vision.capture.base import Capture
from poker_vision.capture.continuity import ContinuityCapture
from poker_vision.capture.frame import Frame
from poker_vision.capture.image_dir import ImageDirCapture
from poker_vision.capture.video_file import VideoFileCapture
from poker_vision.config import SourceConfig, SourceType

__all__ = [
    "Capture",
    "ContinuityCapture",
    "Frame",
    "ImageDirCapture",
    "VideoFileCapture",
    "create_capture",
]


def create_capture(source: SourceConfig) -> Capture:
    """Build the `Capture` implementation selected by `source.type`.

    `source`'s own validator (REQ-2/SourceConfig) already guarantees
    `device_index` is set for `continuity` and `path` is set for the two
    file-backed types, so no redundant check is needed here.
    """
    if source.type is SourceType.CONTINUITY:
        return ContinuityCapture(source.device_index, source.resolution_cap)
    if source.type is SourceType.VIDEO_FILE:
        return VideoFileCapture(source.path, source.resolution_cap)
    return ImageDirCapture(source.path, source.resolution_cap)
