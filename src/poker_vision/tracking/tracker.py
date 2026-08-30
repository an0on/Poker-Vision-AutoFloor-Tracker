"""Nearest-match tracker (REQ-23).

Assigns a stable `track_id` to each detection by nearest-neighbor matching
against the previous known position of every track of the same class, in
table coordinates. This is deliberately simple, own matching logic, not
ByteTrack: ByteTrack is only wired in once the project's own `yolo` model
exists.

Matching is per class (a `chip` never matches a `card`'s track), respects
`max_distance` (`ThresholdsConfig.tracking_max_distance`), and -- per
class -- picks the pairing of tracks to detections that keeps the most
valid (within-threshold) pairs, breaking ties by total distance (see
`matching.optimal_assignment`): a plain greedy "closest pair first" can
strand a detection as a spurious new track even when a valid pairing for
every track exists.

A track's last-known position is kept in memory even for frames where it
isn't matched, so a detection that briefly drops out (e.g. `mock`'s
dropout/occlusion) and reappears near where it left off still recovers its
original `track_id`, rather than starting a new one. Deciding when a track
should instead be treated as gone is REQ-24's hysteresis (`n_off`), not
this stage's job. What this stage does do on its own is evict a track that
has gone unseen for `_STALE_TRACK_TTL_CALLS` calls: a plain growth cap
against unbounded memory (e.g. a steady trickle of ghost detections each
minting a track that then never reappears) and against an old ID drifting
onto an unrelated later object at the same spot -- not REQ-24's hysteresis,
which is a real state machine with configurable, per-class `n_on`/`n_off`
counted in frames, not calls.

`update()` also rejects any detection whose center or box lies outside the
calibrated table (`CalibrationRuntime.table`) before matching runs at all:
that can only mean a bug upstream (a bad homography, a detector that
somehow escaped `Detector.detect()`'s pixel -> table transform), and
matching against it would silently produce meaningless track positions
instead of surfacing the problem.
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict

from poker_vision.calibration.geometry import TableDimensions, TablePoint
from poker_vision.detection.models import Detection, DetectionClass, FrameDetections
from poker_vision.tracking.matching import optimal_assignment
from poker_vision.tracking.models import TRACKING_SCHEMA_VERSION, TrackedFrame, TrackedObject

# Generous relative to any realistic REQ-24 n_off (default 3, config-bounded
# small ints): this is a safety net against unbounded growth, not the
# hysteresis "absent" decision itself.
_STALE_TRACK_TTL_CALLS = 300


def _distance(a: TablePoint, b: TablePoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _check_within_table(point: TablePoint, table: TableDimensions, detection: Detection) -> None:
    if not (0.0 <= point.x <= table.width and 0.0 <= point.y <= table.height):
        raise ValueError(
            f"{detection.object_class.value} detection at ({point.x}, {point.y}) lies outside "
            f"the calibrated table ({table.width}x{table.height} {table.unit.value}); "
            "a detector must never emit table coordinates beyond the table it was calibrated for"
        )


class NearestMatchTracker:
    """Stateful, per-class nearest-neighbor track-ID assignment (REQ-23)."""

    def __init__(self, max_distance: float, table: TableDimensions) -> None:
        self._max_distance = max_distance
        self._table = table
        self._next_track_id = itertools.count(1)
        # Last known table-plane position per class, keyed by track_id.
        self._known: dict[DetectionClass, dict[int, TablePoint]] = defaultdict(dict)
        # `update()` call index (not frame_index -- see module docstring)
        # at which each track was last matched; drives staleness eviction.
        self._last_matched_call: dict[DetectionClass, dict[int, int]] = defaultdict(dict)
        self._call_count = 0

    def update(self, frame_detections: FrameDetections) -> TrackedFrame:
        # Validate before any state changes: a rejected call must be
        # atomic -- no bumped call counter, no eviction -- so retrying with
        # corrected detections sees exactly the state before the bad call,
        # not one that was already (partly) advanced by it.
        for detection in frame_detections.detections:
            _check_within_table(detection.center, self._table, detection)
            if detection.box is not None:
                _check_within_table(detection.box.min, self._table, detection)
                _check_within_table(detection.box.max, self._table, detection)

        self._call_count += 1
        # Evict before matching, not after: a track that just crossed the
        # TTL this call must not be resurrected by a same-call detection
        # landing on its old position -- matching would otherwise refresh
        # `_last_matched_call` first and the eviction sweep below would find
        # nothing left to evict, defeating the point of the TTL.
        self._evict_stale_tracks()

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
        track_ids = list(known.keys())

        valid_cost: dict[tuple[int, int], float] = {}
        for detection_index, detection in enumerate(detections):
            for track_index, track_id in enumerate(track_ids):
                distance = _distance(detection.center, known[track_id])
                if distance <= self._max_distance:
                    valid_cost[(detection_index, track_index)] = distance

        assignment = optimal_assignment(
            row_count=len(detections),
            col_count=len(track_ids),
            valid_cost=valid_cost,
            max_valid_cost=self._max_distance,
        )

        tracks: list[TrackedObject] = []
        # Start from `known` rather than empty: a track of this class that
        # isn't matched this frame (e.g. one of two chips is occluded while
        # the other stays visible) must keep its last position, the same as
        # a class with zero detections this frame keeps all of its tracks.
        updated_known: dict[int, TablePoint] = dict(known)
        last_matched_call = self._last_matched_call[object_class]
        for detection_index, detection in enumerate(detections):
            if detection_index in assignment:
                track_id = track_ids[assignment[detection_index]]
            else:
                track_id = next(self._next_track_id)
            updated_known[track_id] = detection.center
            last_matched_call[track_id] = self._call_count
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

    def _evict_stale_tracks(self) -> None:
        for object_class, known in self._known.items():
            last_matched_call = self._last_matched_call[object_class]
            stale = [
                track_id
                for track_id in known
                if self._call_count - last_matched_call.get(track_id, 0)
                > _STALE_TRACK_TTL_CALLS
            ]
            for track_id in stale:
                del known[track_id]
                last_matched_call.pop(track_id, None)
