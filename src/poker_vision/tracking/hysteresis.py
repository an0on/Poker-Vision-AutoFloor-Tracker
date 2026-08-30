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
per class via `HysteresisConfig.per_class` (keyed by `DetectionClass.value`;
either field may be overridden independently, the other falls back to the
global default).

`n_on`/`n_off` count actual frames, not `update()` calls: `frame_index` is
the authority. Every `Capture`/`Detector` implementation in this project
calls this stage once per captured frame (see e.g. `mock`'s Modus A: "A
frame index with no line in the script yields no detections", not "no
call"), so in normal operation frame indices arrive one at a time with no
gaps. But `TrackedFrame.frame_index` itself carries no such guarantee, and
`NearestMatchTracker` explicitly punts exact frame accounting to this
stage (its own TTL is deliberately keyed on call count, not frame_index --
see its docstring). So `update()` treats a jump in `frame_index` between
two calls as that many frames having silently elapsed with no data at all:
every pending (not-yet-confirmed) track's run is broken, exactly as if it
had been missed that many times, and every confirmed track accrues that
many misses toward `n_off` before the current call's own sightings are
even considered. A non-increasing `frame_index` (equal to or before the
last processed one) can only mean a caller replayed or reordered frames --
that is rejected outright rather than silently reinterpreted.

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

`HysteresisFilter.update()` implements REQ-25 by construction: its return
value only ever contains present tracks, so passing it straight to
`assignment` already excludes anything not yet (or no longer) confirmed.
"""

from __future__ import annotations

from collections import defaultdict

from poker_vision.config import HysteresisConfig
from poker_vision.detection.models import DetectionClass
from poker_vision.tracking.models import TRACKING_SCHEMA_VERSION, TrackedFrame, TrackedObject


class HysteresisFilter:
    """Debounces `NearestMatchTracker` output into confirmed-present tracks."""

    def __init__(self, config: HysteresisConfig) -> None:
        self._config = config
        # Consecutive-sightings count for a track not yet confirmed present.
        # Absent entirely once a track is confirmed (see `_confirmed` below).
        self._on_count: dict[DetectionClass, dict[int, int]] = defaultdict(dict)
        # Consecutive-miss count for a track that *is* confirmed present.
        self._off_count: dict[DetectionClass, dict[int, int]] = defaultdict(dict)
        # Last known TrackedObject for every currently-confirmed track;
        # carried forward on frames where it's missing but not yet expired.
        self._confirmed: dict[DetectionClass, dict[int, TrackedObject]] = defaultdict(dict)
        # frame_index of the last processed call; None before the first one.
        self._last_frame_index: int | None = None

    def update(self, tracked_frame: TrackedFrame) -> TrackedFrame:
        frame_index = tracked_frame.frame_index
        if self._last_frame_index is None:
            skipped_frames = 0
        elif frame_index <= self._last_frame_index:
            raise ValueError(
                f"HysteresisFilter.update() received frame_index {frame_index}, which is "
                f"not after the last processed frame_index {self._last_frame_index}; frames "
                "must be applied in strictly increasing order so n_on/n_off can be counted "
                "in actual frames"
            )
        else:
            skipped_frames = frame_index - self._last_frame_index - 1

        seen_by_class: dict[DetectionClass, dict[int, TrackedObject]] = defaultdict(dict)
        for track in tracked_frame.tracks:
            seen_by_class[track.object_class][track.track_id] = track

        known_classes = set(self._confirmed) | set(self._on_count) | set(seen_by_class)
        stable: list[TrackedObject] = []
        for object_class in known_classes:
            stable.extend(
                self._update_class(
                    object_class, seen_by_class.get(object_class, {}), skipped_frames
                )
            )

        self._last_frame_index = frame_index
        return TrackedFrame(
            schema_version=TRACKING_SCHEMA_VERSION,
            frame_index=frame_index,
            tracks=stable,
        )

    def _thresholds(self, object_class: DetectionClass) -> tuple[int, int]:
        override = self._config.per_class.get(object_class.value)
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
    ) -> list[TrackedObject]:
        n_on, n_off = self._thresholds(object_class)
        on_count = self._on_count[object_class]
        off_count = self._off_count[object_class]
        confirmed = self._confirmed[object_class]

        if skipped_frames > 0:
            # `skipped_frames` frames elapsed with no call at all for this
            # class: every pending track's run is broken (a real gap, not
            # just "not in this call's seen set"), and every confirmed
            # track accrues that many misses toward n_off before this
            # call's own sightings are considered at all.
            on_count.clear()
            for track_id in list(confirmed):
                count = off_count.get(track_id, 0) + skipped_frames
                if count >= n_off:
                    del confirmed[track_id]
                    off_count.pop(track_id, None)
                else:
                    off_count[track_id] = count

        stable: list[TrackedObject] = []
        for track_id, track in seen.items():
            if track_id in confirmed:
                confirmed[track_id] = track
                off_count.pop(track_id, None)
                stable.append(track)
                continue
            count = on_count.get(track_id, 0) + 1
            if count >= n_on:
                confirmed[track_id] = track
                on_count.pop(track_id, None)
                stable.append(track)
            else:
                on_count[track_id] = count

        # A pending (not-yet-confirmed) track missing this frame breaks its
        # run of consecutive sightings -- drop it rather than let it resume
        # counting from where it left off.
        for track_id in list(on_count):
            if track_id not in seen:
                del on_count[track_id]

        # A confirmed track missing this frame moves toward `n_off`; below
        # that, it stays present via its carried-forward last-known state.
        for track_id in list(confirmed):
            if track_id in seen:
                continue
            count = off_count.get(track_id, 0) + 1
            if count >= n_off:
                del confirmed[track_id]
                off_count.pop(track_id, None)
            else:
                off_count[track_id] = count
                stable.append(confirmed[track_id])

        return stable
