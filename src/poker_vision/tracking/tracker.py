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
this stage's job. What this stage does do on its own, as a plain safety
net (not REQ-24's hysteresis, which is a real state machine with
configurable, per-class `n_on`/`n_off` counted in frames, not calls) is
bound how much of this class's history can pile up:

- Evict a track that has gone unseen for `stale_track_ttl` calls (against
  an old ID drifting onto an unrelated later object at the same spot).
  This is a constructor parameter, not a hardcoded constant, precisely
  because it must not silently outrank a real `n_off`: `HysteresisConfig.
  n_off` has no configured upper bound, so whoever wires REQ-24's
  hysteresis on top of this tracker needs to pass a `stale_track_ttl` at
  least as large as the largest `n_off` in play (across the global value
  and every per-class override) -- otherwise this safety net could retire
  a track's ID before hysteresis ever gets to decide it's actually gone.
  `_DEFAULT_STALE_TRACK_TTL_CALLS` is only a reasonable value for when no
  such config exists yet (as in this REQ's own tests).
- Cap each class's remembered tracks at `_MAX_KNOWN_TRACKS_PER_CLASS`,
  evicting the least-recently-matched ones first whenever a class would
  exceed it -- checked every call, not just once the TTL has had time to
  bite. Without this, a run of detections that never re-match anything
  (e.g. sustained `mock` ghost/jitter beyond the matching threshold, or
  simply a lively table) can grow one class's remembered-track count far
  past this frame's own detection count within a handful of calls, long
  before the TTL would ever trim it -- and `matching.optimal_assignment`
  is O((detections + known tracks)^3), so an uncapped few hundred known
  tracks alone (measured, pure Python) already costs whole seconds per
  frame, blowing REQ-42's per-frame budget. `_MAX_KNOWN_TRACKS_PER_CLASS`
  reuses REQ-42's own "<=50 detections/frame" ceiling as the cap, so a
  single class's assignment problem never exceeds roughly detections +
  50 candidates, regardless of how detections behave over time.

Before even building the candidate list, tracks with no within-threshold
detection this frame are also excluded from the columns handed to
`optimal_assignment`: excluding a candidate that can never be picked
changes nothing about the result, and keeps the typical-case problem size
driven by genuine spatial proximity rather than the capped history's full
size.

`update()` also rejects any detection whose *center* lies outside the
calibrated table (`CalibrationRuntime.table`) before matching runs at all:
that can only mean a bug upstream (a bad homography, a detector that
somehow escaped `Detector.detect()`'s pixel -> table transform), and
matching against it would silently produce meaningless track positions
instead of surfacing the problem. A detection's optional `box` is *not*
checked against the table: REQ-17 only requires the box to be in table
coordinates, not contained within the table, and a real detector (e.g.
`mock`'s COCO mode) can legitimately report a box that straddles the
table edge for an object sitting right at the rim -- that is normal
output, not a bug, and tracking only ever matches on `center` anyway.

REQ-44's core-chain commit policy ("jede Stufe berechnet ihr Update rein,
ohne eigenen persistenten Zustand direkt zu mutieren") splits what used to
be one mutating `update()` into `compute_update()` (pure -- reads `self`,
never writes it) and `commit()` (applies a previously computed
`TrackerUpdate` to `self`). `update()` itself is kept as the two called
back-to-back, so every other caller (this module's own tests included)
sees the exact same mutate-immediately behavior as before; only the
runner's frame loop needs the two steps split apart, to defer this
stage's mutation until the whole core chain (tracking -> assignment ->
state) has succeeded for the frame (see `runner/loop.py`).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from poker_vision.calibration.geometry import TableDimensions, TablePoint
from poker_vision.detection.models import Detection, DetectionClass, FrameDetections
from poker_vision.tracking.matching import optimal_assignment
from poker_vision.tracking.models import TRACKING_SCHEMA_VERSION, TrackedFrame, TrackedObject

# Default for `NearestMatchTracker(stale_track_ttl=...)`: generous relative
# to HysteresisConfig's own default n_off (3), but this is only a sane
# fallback for when no hysteresis config is wired up yet -- see the
# constructor parameter's docstring for why a real n_off must be passed in
# explicitly once REQ-24 exists.
_DEFAULT_STALE_TRACK_TTL_CALLS = 30

# Reuses REQ-42's own "<=50 detections/frame" ceiling: a single class's
# remembered-track count never needs to exceed what one frame could ever
# contain. Enforced every call (not just once the TTL expires) via
# least-recently-matched eviction, so `optimal_assignment`'s per-class
# problem size never exceeds roughly this many candidates plus the
# current frame's own detection count in that class.
_MAX_KNOWN_TRACKS_PER_CLASS = 50

_KnownTracks = dict[DetectionClass, dict[int, TablePoint]]
_LastMatchedCall = dict[DetectionClass, dict[int, int]]


@dataclass(frozen=True, slots=True)
class TrackerUpdate:
    """Pure result of `NearestMatchTracker.compute_update()`.

    Carries this call's output (`tracked_frame`) together with the
    would-be new internal state (`known`, `last_matched_call`,
    `call_count`, `next_track_id`) so `commit()` can apply it verbatim,
    without recomputing anything and without `compute_update()` having
    touched `self` at all.
    """

    tracked_frame: TrackedFrame
    known: _KnownTracks
    last_matched_call: _LastMatchedCall
    call_count: int
    next_track_id: int


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
    """Per-class nearest-neighbor track-ID assignment (REQ-23).

    `stale_track_ttl` (calls, not frames -- see `compute_update()`'s
    docstring) must be at least as large as the largest `n_off` a caller
    intends to honor once REQ-24's hysteresis is wired on top of this
    tracker: a track this tracker has already forgotten can never recover
    its ID, no matter what hysteresis would have decided. Defaults to
    `_DEFAULT_STALE_TRACK_TTL_CALLS`, a reasonable value only for when no
    such config exists yet.
    """

    def __init__(
        self,
        max_distance: float,
        table: TableDimensions,
        stale_track_ttl: int = _DEFAULT_STALE_TRACK_TTL_CALLS,
    ) -> None:
        self._max_distance = max_distance
        self._table = table
        self._stale_track_ttl = stale_track_ttl
        self._next_track_id = 1
        # Last known table-plane position per class, keyed by track_id.
        self._known: _KnownTracks = {}
        # `compute_update()` call index (not frame_index -- see module
        # docstring) at which each track was last matched; drives
        # staleness eviction.
        self._last_matched_call: _LastMatchedCall = {}
        self._call_count = 0

    def compute_update(self, frame_detections: FrameDetections) -> TrackerUpdate:
        """Pure computation of this call's tracks and the resulting state.

        Never mutates `self` -- see module docstring. Validates before
        computing anything: a rejected call must be atomic, so a caller
        retrying with corrected detections sees exactly the state before
        the bad call, not a partially-advanced one. Only `center` is
        checked (see module docstring): a box may legitimately extend
        past the table edge.
        """
        for detection in frame_detections.detections:
            _check_within_table(detection.center, self._table, detection)

        call_count = self._call_count + 1
        known: _KnownTracks = {cls: dict(tracks) for cls, tracks in self._known.items()}
        last_matched_call: _LastMatchedCall = {
            cls: dict(calls) for cls, calls in self._last_matched_call.items()
        }
        # Evict before matching, not after: a track that just crossed the
        # TTL this call must not be resurrected by a same-call detection
        # landing on its old position -- matching would otherwise refresh
        # `last_matched_call` first and the eviction sweep below would
        # find nothing left to evict, defeating the point of the TTL.
        self._evict_stale_tracks(known, last_matched_call, call_count)

        by_class: dict[DetectionClass, list[Detection]] = defaultdict(list)
        for detection in frame_detections.detections:
            by_class[detection.object_class].append(detection)

        next_track_id = self._next_track_id
        tracks: list[TrackedObject] = []
        for object_class, detections in by_class.items():
            class_tracks, next_track_id = self._match_class(
                object_class, detections, known, last_matched_call, call_count, next_track_id
            )
            tracks.extend(class_tracks)

        return TrackerUpdate(
            tracked_frame=TrackedFrame(
                schema_version=TRACKING_SCHEMA_VERSION,
                frame_index=frame_detections.frame_index,
                tracks=tracks,
            ),
            known=known,
            last_matched_call=last_matched_call,
            call_count=call_count,
            next_track_id=next_track_id,
        )

    def commit(self, update: TrackerUpdate) -> TrackedFrame:
        """Apply a previously computed `TrackerUpdate` to `self`."""
        self._known = update.known
        self._last_matched_call = update.last_matched_call
        self._call_count = update.call_count
        self._next_track_id = update.next_track_id
        return update.tracked_frame

    def update(self, frame_detections: FrameDetections) -> TrackedFrame:
        """Compute and immediately commit this call's update (see module docstring)."""
        return self.commit(self.compute_update(frame_detections))

    def _match_class(
        self,
        object_class: DetectionClass,
        detections: list[Detection],
        known: _KnownTracks,
        last_matched_call: _LastMatchedCall,
        call_count: int,
        next_track_id: int,
    ) -> tuple[list[TrackedObject], int]:
        class_known = known.get(object_class, {})

        # Only tracks with at least one within-threshold detection this
        # frame can ever be picked by `optimal_assignment`; dropping the
        # rest here changes nothing about the result but keeps the typical
        # -case problem size driven by spatial proximity, not this class's
        # entire (capped) remembered history.
        cost_by_track_id: dict[int, dict[int, float]] = defaultdict(dict)
        for detection_index, detection in enumerate(detections):
            for track_id, position in class_known.items():
                distance = _distance(detection.center, position)
                if distance <= self._max_distance:
                    cost_by_track_id[track_id][detection_index] = distance
        track_ids = list(cost_by_track_id.keys())

        valid_cost: dict[tuple[int, int], float] = {
            (detection_index, track_index): distance
            for track_index, track_id in enumerate(track_ids)
            for detection_index, distance in cost_by_track_id[track_id].items()
        }

        assignment = optimal_assignment(
            row_count=len(detections),
            col_count=len(track_ids),
            valid_cost=valid_cost,
            max_valid_cost=self._max_distance,
        )

        tracks: list[TrackedObject] = []
        # Start from `class_known` rather than empty: a track of this class
        # that isn't matched this frame (e.g. one of two chips is occluded
        # while the other stays visible) must keep its last position, the
        # same as a class with zero detections this frame keeps all of its
        # tracks.
        updated_known: dict[int, TablePoint] = dict(class_known)
        class_last_matched_call = last_matched_call.setdefault(object_class, {})
        for detection_index, detection in enumerate(detections):
            if detection_index in assignment:
                track_id = track_ids[assignment[detection_index]]
            else:
                track_id = next_track_id
                next_track_id += 1
            updated_known[track_id] = detection.center
            class_last_matched_call[track_id] = call_count
            tracks.append(
                TrackedObject(
                    track_id=track_id,
                    object_class=object_class,
                    confidence=detection.confidence,
                    center=detection.center,
                    box=detection.box,
                )
            )

        known[object_class] = updated_known
        self._enforce_known_track_cap(object_class, known, last_matched_call)
        return tracks, next_track_id

    def _enforce_known_track_cap(
        self, object_class: DetectionClass, known: _KnownTracks, last_matched_call: _LastMatchedCall
    ) -> None:
        class_known = known[object_class]
        overflow = len(class_known) - _MAX_KNOWN_TRACKS_PER_CLASS
        if overflow <= 0:
            return
        class_last_matched_call = last_matched_call.setdefault(object_class, {})
        # Evict the least-recently-matched tracks first: within a single
        # over-cap frame they're all equally "fresh" by call count, so ties
        # fall back to `class_known`'s insertion order (oldest-created first).
        least_recent = sorted(
            class_known, key=lambda track_id: class_last_matched_call.get(track_id, 0)
        )
        for track_id in least_recent[:overflow]:
            del class_known[track_id]
            class_last_matched_call.pop(track_id, None)

    def _evict_stale_tracks(
        self, known: _KnownTracks, last_matched_call: _LastMatchedCall, call_count: int
    ) -> None:
        for object_class, class_known in known.items():
            class_last_matched_call = last_matched_call.setdefault(object_class, {})
            stale = [
                track_id
                for track_id in class_known
                if call_count - class_last_matched_call.get(track_id, 0) > self._stale_track_ttl
            ]
            for track_id in stale:
                del class_known[track_id]
                class_last_matched_call.pop(track_id, None)
