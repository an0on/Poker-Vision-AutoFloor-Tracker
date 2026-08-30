"""`image_dir` capture: deterministic frame sequence from a directory of
still images, no camera involved (REQ-13, REQ-15).
"""

from __future__ import annotations

from pathlib import Path

import cv2

from poker_vision.capture.base import Capture
from poker_vision.capture.frame import Frame
from poker_vision.capture.resolution import apply_resolution_cap
from poker_vision.capture.timestamps import replay_timestamp
from poker_vision.config import Resolution

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# No inherent frame rate exists for a directory of stills; this only fixes
# the spacing between synthetic timestamps, which is otherwise arbitrary
# (see `timestamps.replay_timestamp`).
_SYNTHETIC_FPS = 1.0


class ImageDirCapture(Capture):
    """Yields one `Frame` per image file in `path`, sorted by filename.

    Sorting by filename (rather than e.g. mtime) is what makes REQ-15's
    "identical input -> identical sequence" hold on any filesystem/checkout.
    """

    def __init__(
        self,
        path: str | Path,
        resolution_cap: Resolution,
        source_id: str | None = None,
    ) -> None:
        self._dir = Path(path)
        self._paths = sorted(
            p for p in self._dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not self._paths:
            raise ValueError(f"no images found in {self._dir}")
        self._resolution_cap = resolution_cap
        self._index = 0
        self.source_id = source_id or f"image_dir:{self._dir}"

    def __next__(self) -> Frame:
        if self._index >= len(self._paths):
            raise StopIteration
        image_path = self._paths[self._index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        image = apply_resolution_cap(image, self._resolution_cap)
        frame = Frame(
            image=image,
            timestamp=replay_timestamp(self._index, _SYNTHETIC_FPS),
            frame_index=self._index,
            source_id=self.source_id,
        )
        self._index += 1
        return frame

    def close(self) -> None:
        pass
