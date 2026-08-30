"""Presence hysteresis on top of per-frame tracking (REQ-24, REQ-25).

`NearestMatchTracker` (REQ-23) reports, every frame, exactly the tracks it
matched a detection to that frame -- a single missed frame (occlusion,
detector noise) makes a track vanish from its output, and a single stray
detection (ghost) makes one appear. Neither is fit to drive state changes
like `seat_occupied`/`seat_vacated` directly: REQ-24 debounces both edges
per track, independently per class:

- A track counts as *present* only after `n_on` **consecutive** frames with
  a match. A miss before reaching `n_on` resets that track's count to zero
  -- a ghost that never strings together `n_on` consecutive sightings never
  becomes present at all (AC-12's "Geister-Detection < n_on Frames erzeugt
  kein seat_occupied").
- Once present, a track counts as *absent* only after `n_off` consecutive
  missed frames. A track that reappears before reaching `n_off` stays
  present the whole time, using its last known position/confidence/box for
  the frames it was missing -- assignment (REQ-26) still needs a position
  to test against a zone, and the track's own tracker state may have moved
  on in the meantime (AC-12's "Dropout < n_off Frames entsteht kein
  seat_vacated").

`n_on`/`n_off` come from `HysteresisConfig`: a global default, overridable
per class via `HysteresisConfig.per_class` (keyed by `DetectionClass`, so
config loading itself rejects an unsupported or typo'd class rather than
that override silently never matching anything; either field may be
overridden independently, the other falls back to the global default).

`n_on`/`n_off` count actual frames, not `compute_update()` calls:
`frame_index` is the authority. Every `Capture`/`Detector` implementation
in this project calls this stage once per captured frame (see e.g.
`mock`'s Modus A: "A frame index with no line in the script yields no
detections", not "no call"), so in normal operation frame indices arrive
one at a time with no gaps. But `TrackedFrame.frame_index` itself carries
no such guarantee, and `NearestMatchTracker` explicitly punts exact frame
accounting to this stage (its own TTL is deliberately keyed on call count,
not frame_index -- see its docstring). So `compute_update()` treats a jump
in `frame_index` between two calls as that many frames having silently
elapsed with no data at all: every pending (not-yet-confirmed) track's run
is broken, exactly as if it had been missed that many times, and every
confirmed track accrues that many misses toward `n_off` before the current
call's own sightings are even considered. A non-increasing `frame_index`
(equal to or before the last processed one) can only mean a caller
replayed or reordered frames -- that is rejected outright rather than
silently reinterpreted.

This is a separate stage from `NearestMatchTracker`, not a parameter to it:
it owns its own per-(class, track_id) present/absent state, decoupled from
the tracker's internal matching state (which tracks a different, shorter-
lived question -- "is this still the same physical object" -- via its own
`stale_track_ttl` safety net, not this frame-counted state machine). A
pipeline wiring both together must give the tracker a `stale_track_ttl` at
least as large as the largest configured `n_off` (global and every
per-class override): a track ID this filter is still waiting out `n_off`
missed frames for must not be evicted -- and thus never reusable if it
reappears -- by the tracker first. See `NearestMatchTracker`'s own
docstring for the same requirement from its side.

`HysteresisFilter`'s output only ever contains present tracks (REQ-25):
passing it straight to `assignment` already excludes anything not yet (or
no longer) confirmed.

REQ-44's core-chain commit policy splits what used to be one mutating
`update()` into `compute_update()` (pure) and `commit()` (applies a
previously computed `HysteresisUpdate`). `update()` is kept as the two
called back-to-back, for standalone callers (this module's own tests
included); the runner's frame loop uses the two steps separately, to defer
this stage's mutation until the whole core chain has succeeded for the
frame (see `runner/loop.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

from poker_vision.config import HysteresisConfig
from poker_vision.detection.models import DetectionClass
from poker_vision.tracking.models import TRACKING_SCHEMA_VERSION, TrackedFrame, TrackedObject

_OnCount = dict[DetectionClass, dict[int, int]]
_OffCount = dict[DetectionClass, dict[int, int]]
_Confirmed = dict[DetectionClass, dict[int, TrackedObject]]


@dataclass(frozen=True, slots=True)
class HysteresisUpdate:
    """Pure result of `HysteresisFilter.compute_update()`.

    Carries this call's output (`tracked_frame`, the confirmed-present
    tracks) together with the would-be new internal state so `commit()`
    can apply it verbatim.
    """

    tracked_frame: TrackedFrame
    on_count: _OnCount
    off_count: _OffCount
    confirmed: _Confirmed
    frame_index: int


class HysteresisFilter:
    """Debounces `NearestMatchTracker` output into confirmed-present tracks."""

    def __init__(self, config: HysteresisConfig) -> None:
        self._config = config
        # Consecutive-sightings count for a track not yet confirmed present.
        # Absent entirely once a track is confirmed (see `_confirmed` below).
        self._on_count: _OnCount = {}
        # Consecutive-miss count for a track that *is* confirmed present.
        self._off_count: _OffCount = {}
        # Last known TrackedObject for every currently-confirmed track;
        # carried forward on frames where it's missing but not yet expired.
        self._confirmed: _Confirmed = {}
        # frame_index of the last processed call; None before the first one.
        self._last_frame_index: int | None = None

    def compute_update(self, tracked_frame: TrackedFrame) -> HysteresisUpdate:
        """Pure computation of this call's stable tracks and the resulting state.

        Never mutates `self` -- see module docstring.
        """
        frame_index = tracked_frame.frame_index
        if self._last_frame_index is None:
            skipped_frames = 0
        elif frame_index <= self._last_frame_index:
            raise ValueError(
                f"HysteresisFilter.compute_update() received frame_index {frame_index}, which "
                f"is not after the last processed frame_index {self._last_frame_index}; frames "
                "must be applied in strictly increasing order so n_on/n_off can be counted "
                "in actual frames"
            )
        else:
            skipped_frames = frame_index - self._last_frame_index - 1

        seen_by_class: dict[DetectionClass, dict[int, TrackedObject]] = {}
        for track in tracked_frame.tracks:
            seen_by_class.setdefault(track.object_class, {})[track.track_id] = track

        on_count: _OnCount = {cls: dict(counts) for cls, counts in self._on_count.items()}
        off_count: _OffCount = {cls: dict(counts) for cls, counts in self._off_count.items()}
        confirmed: _Confirmed = {cls: dict(tracks) for cls, tracks in self._confirmed.items()}

        known_classes = set(confirmed) | set(on_count) | set(seen_by_class)
        stable: list[TrackedObject] = []
        for object_class in known_classes:
            stable.extend(
                self._update_class(
                    object_class,
                    seen_by_class.get(object_class, {}),
                    skipped_frames,
                    on_count,
                    off_count,
                    confirmed,
                )
            )

        return HysteresisUpdate(
            tracked_frame=TrackedFrame(
                schema_version=TRACKING_SCHEMA_VERSION, frame_index=frame_index, tracks=stable
            ),
            on_count=on_count,
            off_count=off_count,
            confirmed=confirmed,
            frame_index=frame_index,
        )

    def commit(self, update: HysteresisUpdate) -> TrackedFrame:
        """Apply a previously computed `HysteresisUpdate` to `self`."""
        self._on_count = update.on_count
        self._off_count = update.off_count
        self._confirmed = update.confirmed
        self._last_frame_index = update.frame_index
        return update.tracked_frame

    def update(self, tracked_frame: TrackedFrame) -> TrackedFrame:
        """Compute and immediately commit this call's update (see module docstring)."""
        return self.commit(self.compute_update(tracked_frame))

    def _thresholds(self, object_class: DetectionClass) -> tuple[int, int]:
        override = self._config.per_class.get(object_class)
        n_on = self._config.n_on if override is None or override.n_on is None else override.n_on
        n_off = (
            self._config.n_off if override is None or override.n_off is None else override.n_off
        )
        return n_on, n_off

    def _update_class(
        self,
        object_class: DetectionClass,
        seen: dict[int, TrackedObject],
        skipped_frames: int,
        on_count: _OnCount,
        off_count: _OffCount,
        confirmed: _Confirmed,
    ) -> list[TrackedObject]:
        n_on, n_off = self._thresholds(object_class)
        class_on_count = on_count.setdefault(object_class, {})
        class_off_count = off_count.setdefault(object_class, {})
        class_confirmed = confirmed.setdefault(object_class, {})

        if skipped_frames > 0:
            # `skipped_frames` frames elapsed with no call at all for this
            # class: every pending track's run is broken (a real gap, not
            # just "not in this call's seen set"), and every confirmed
            # track accrues that many misses toward n_off before this
            # call's own sightings are considered at all.
            class_on_count.clear()
            for track_id in list(class_confirmed):
                count = class_off_count.get(track_id, 0) + skipped_frames
                if count >= n_off:
                    del class_confirmed[track_id]
                    class_off_count.pop(track_id, None)
                else:
                    class_off_count[track_id] = count

        stable: list[TrackedObject] = []
        for track_id, track in seen.items():
            if track_id in class_confirmed:
                class_confirmed[track_id] = track
                class_off_count.pop(track_id, None)
                stable.append(track)
                continue
            count = class_on_count.get(track_id, 0) + 1
            if count >= n_on:
                class_confirmed[track_id] = track
                class_on_count.pop(track_id, None)
                stable.append(track)
            else:
                class_on_count[track_id] = count

        # A pending (not-yet-confirmed) track missing this frame breaks its
        # run of consecutive sightings -- drop it rather than let it resume
        # counting from where it left off.
        for track_id in list(class_on_count):
            if track_id not in seen:
                del class_on_count[track_id]

        # A confirmed track missing this frame moves toward `n_off`; below
        # that, it stays present via its carried-forward last-known state.
        for track_id in list(class_confirmed):
            if track_id in seen:
                continue
            count = class_off_count.get(track_id, 0) + 1
            if count >= n_off:
                del class_confirmed[track_id]
                class_off_count.pop(track_id, None)
            else:
                class_off_count[track_id] = count
                stable.append(class_confirmed[track_id])

        return stable
