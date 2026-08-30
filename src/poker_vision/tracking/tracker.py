"""Nearest-match tracker (REQ-23).

Assigns a stable `track_id` to each detection by nearest-neighbor matching
against the previous known position of every track of the same class, in
table coordinates. This is deliberately simple, own matching logic, not
ByteTrack: ByteTrack is only wired in once the project's own `yolo` model
exists.

Matching is per class (a `chip` never matches a `card`'s track) and
respects `max_distance` (`ThresholdsConfig.tracking_max_distance`): a
detection farther than that from every known track of its class starts a
new track instead of reusing one.

A track's last-known position is kept in memory even for frames where it
isn't matched, so a detection that briefly drops out (e.g. `mock`'s
dropout/occlusion) and reappears near where it left off still recovers its
original `track_id`, rather than starting a new one. Deciding when a track
should instead be treated as gone is REQ-24's hysteresis (`n_off`), not
this stage's job — this tracker never expires a track on its own.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

from poker_vision.calibration.geometry import TablePoint
from poker_vision.detection.models import Detection, DetectionClass, FrameDetections
from poker_vision.tracking.models import TRACKING_SCHEMA_VERSION, TrackedFrame, TrackedObject


def _distance(a: TablePoint, b: TablePoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


class NearestMatchTracker:
    """Stateful, per-class nearest-neighbor track-ID assignment (REQ-23)."""

    def __init__(self, max_distance: float) -> None:
        self._max_distance = max_distance
        self._next_track_id = itertools.count(1)
        # Last known table-plane position per class, keyed by track_id.
        self._known: dict[DetectionClass, dict[int, TablePoint]] = defaultdict(dict)

    def update(self, frame_detections: FrameDetections) -> TrackedFrame:
        by_class: dict[DetectionClass, list[Detection]] = defaultdict(list)
        for detection in frame_detections.detections:
            by_class[detection.object_class].append(detection)

        tracks = [
            track
            for object_class, detections in by_class.items()
            for track in self._match_class(object_class, detections)
        ]
        return TrackedFrame(
            schema_version=TRACKING_SCHEMA_VERSION,
            frame_index=frame_detections.frame_index,
            tracks=tracks,
        )

    def _match_class(
        self, object_class: DetectionClass, detections: list[Detection]
    ) -> list[TrackedObject]:
        known = self._known[object_class]

        candidates = sorted(
            (
                (_distance(detection.center, center), track_id, detection_index)
                for detection_index, detection in enumerate(detections)
                for track_id, center in known.items()
            ),
            key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
        )

        assigned_track_id: dict[int, int] = {}
        used_track_ids: set[int] = set()
        for distance, track_id, detection_index in candidates:
            if distance > self._max_distance:
                break
            if track_id in used_track_ids or detection_index in assigned_track_id:
                continue
            assigned_track_id[detection_index] = track_id
            used_track_ids.add(track_id)

        tracks: list[TrackedObject] = []
        # Start from `known` rather than empty: a track of this class that
        # isn't matched this frame (e.g. one of two chips is occluded while
        # the other stays visible) must keep its last position, the same as
        # a class with zero detections this frame keeps all of its tracks.
        updated_known: dict[int, TablePoint] = dict(known)
        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_track_id:
                track_id = assigned_track_id[detection_index]
            else:
                track_id = next(self._next_track_id)
            updated_known[track_id] = detection.center
            tracks.append(
                TrackedObject(
                    track_id=track_id,
                    object_class=object_class,
                    confidence=detection.confidence,
                    center=detection.center,
                    box=detection.box,
                )
            )

        self._known[object_class] = updated_known
        return tracks
