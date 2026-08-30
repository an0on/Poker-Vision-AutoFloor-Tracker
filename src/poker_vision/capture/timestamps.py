"""Deterministic timestamp derivation for replay sources (REQ-15).

`video_file`/`image_dir` must yield identical timestamp *ordering* across
repeated runs (AC-8). Deriving the timestamp purely from `frame_index` and a
fixed frame rate — instead of the wall clock — makes replay timestamps fully
reproducible, not just correctly ordered.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

# Arbitrary fixed epoch: only relative order/spacing matters for replay
# sources, never the absolute wall-clock value.
REPLAY_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def replay_timestamp(frame_index: int, fps: float) -> datetime:
    return REPLAY_EPOCH + timedelta(seconds=frame_index / fps)
