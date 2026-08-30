"""The single output shape every `Capture` implementation produces (REQ-13).

`Frame` is a plain runtime dataclass, not a `StrictModel` (REQ-4): it carries
a raw `numpy.ndarray` image buffer, which is not JSON-serializable schema
data — REQ-4's schema list covers calibration, config, detections, events
and the state snapshot, not the pixel buffers flowing between pipeline
stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """One frame, identical in shape regardless of which source produced it.

    `image` is BGR (OpenCV convention), already scaled to the configured
    resolution cap (REQ-14). `frame_index` is a running, zero-based counter
    local to the capture instance. `source_id` identifies which source and
    concrete instance (e.g. device index or file path) produced the frame.
    """

    image: np.ndarray
    timestamp: datetime
    frame_index: int
    source_id: str
