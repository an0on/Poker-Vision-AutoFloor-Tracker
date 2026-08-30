"""Resolution-cap resizing shared by all `Capture` implementations (REQ-14).

Applied uniformly regardless of source, so `video_file`/`image_dir` replay
frames match the same inference resolution a live `continuity` frame would
have.
"""

from __future__ import annotations

import cv2
import numpy as np

from poker_vision.config import Resolution


def apply_resolution_cap(image: np.ndarray, cap: Resolution) -> np.ndarray:
    """Downscale `image` to fit within `cap`, preserving aspect ratio.

    Only ever shrinks (AC-9: "bei Eingabe > Cap"); an image already within
    the cap on both axes is returned unchanged, never upscaled.
    """
    height, width = image.shape[:2]
    scale = min(cap.width / width, cap.height / height, 1.0)
    if scale >= 1.0:
        return image
    new_size = (max(round(width * scale), 1), max(round(height * scale), 1))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
