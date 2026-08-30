"""Frame-loop orchestration (REQ-44).

`FrameLoop` drives one frame at a time through the fixed chain
`capture -> detection -> tracking -> assignment -> state -> export -> debug`,
per the error policy in CLAUDE.md's "Pipeline-Runner" section:

- `detection`/`tracking`/`assignment`/`state` are the "Kernkette": an
  exception in any of them discards the whole frame with no partial
  update, and after `max_consecutive_core_errors` such failures in a row
  the loop aborts (`FatalPipelineError`). A single successful frame resets
  that counter.
- `export` is isolated by `ExportManager` itself and never raises here;
  `debug` is best-effort and wrapped in its own try/except so a rendering
  failure never touches the loop.
- Calibration is not loaded here at all -- REQ-45's lifecycle loads and
  validates it once, fail-fast, before ever constructing a `FrameLoop`;
  this module only ever *applies* the already-validated
  `CalibrationRuntime` it's constructed with.

The commit-after-success rule ("jede Stufe berechnet ihr Update rein,
... der Loop committet ... erst, nachdem die gesamte Kernkette ... erfolgreich
war") is why `tracking` (`NearestMatchTracker`, `HysteresisFilter`) and
`state` (`PipelineStateMachine`) each expose a `compute_update()`/`commit()`
pair instead of one mutating `update()`: `process_frame()` below calls
every stage's `compute_update()` (or, for `detection`/`assignment`, its
one existing pure call -- neither stage holds mutable state that
survives across frames) inside a single `try`, and only calls `commit()`
on `tracker`/`hysteresis`/`state_machine` -- in that pipeline order --
once every one of those calls has returned without raising. An exception
partway through (e.g. `assignment` or `state` failing after `tracking`
already computed a valid update) means `tracker.commit()` is simply never
reached for that frame, so `NearestMatchTracker`'s and `HysteresisFilter`'s
internal state stays exactly as it was before the call.

No pipeline stage imports this module or `FrameContext` (`runner ->
Stufen` is the only allowed dependency direction) -- every stage is still
called through its own existing, typed signature; `FrameContext` is purely
this module's own bookkeeping, assembled from each call's return value.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum

from poker_vision.assignment.zone_assignment import apply_dealer_nearest_seat_fallback, assign_zones
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.base import Capture
from poker_vision.capture.frame import Frame
from poker_vision.config import RunnerConfig
from poker_vision.debug.mjpeg import MjpegDebugServer
from poker_vision.detection.base import Detector
from poker_vision.export.manager import ExportManager
from poker_vision.runner.context import FrameContext
from poker_vision.state.machine import PipelineStateMachine
from poker_vision.tracking.hysteresis import HysteresisFilter
from poker_vision.tracking.tracker import NearestMatchTracker

logger = logging.getLogger(__name__)

# Single source of truth for the "Default 30" in REQ-44's AC: `RunnerConfig`
# is what a real run's `max_consecutive_core_errors` actually comes from
# (REQ-2, "kein Modul liest ... Konstanten direkt"); this default only
# exists so a `FrameLoop` built without a `Config` (e.g. this module's own
# tests) still gets the same number.
_DEFAULT_MAX_CONSECUTIVE_CORE_ERRORS = RunnerConfig().max_consecutive_core_errors


class LoopExitReason(StrEnum):
    """Why `FrameLoop.run()` returned instead of raising."""

    EOF = "eof"


class FatalPipelineError(RuntimeError):
    """Raised by `FrameLoop.run()` once too many core-chain frames in a row failed.

    Distinct from an unhandled exception propagating out of `capture`
    (e.g. a live `continuity` read failure), so a caller (REQ-45's
    lifecycle/CLI) can tell the two fatal conditions apart if it wants to,
    while treating both as "abort with a non-zero exit code".
    """


class FrameLoop:
    """Orchestrates one full pass of the pipeline per frame (REQ-44).

    Every constructor argument is an already-constructed stage instance
    (dependency injection, per the `runner -> Stufen` direction): building
    the concrete stages from a `Config` -- selecting the detector
    implementation, opening the capture source, loading calibration -- is
    the CLI/lifecycle's job (REQ-45), not this class's.
    """

    def __init__(
        self,
        capture: Capture,
        detector: Detector,
        tracker: NearestMatchTracker,
        hysteresis: HysteresisFilter,
        calibration: CalibrationRuntime,
        dealer_nearest_seat_max_distance: float,
        state_machine: PipelineStateMachine,
        export_manager: ExportManager,
        debug_server: MjpegDebugServer | None = None,
        max_consecutive_core_errors: int = _DEFAULT_MAX_CONSECUTIVE_CORE_ERRORS,
        on_frame_processed: Callable[[FrameContext], None] | None = None,
    ) -> None:
        self._capture = capture
        self._detector = detector
        self._tracker = tracker
        self._hysteresis = hysteresis
        self._calibration = calibration
        self._dealer_nearest_seat_max_distance = dealer_nearest_seat_max_distance
        self._state_machine = state_machine
        self._export_manager = export_manager
        self._debug_server = debug_server
        self._max_consecutive_core_errors = max_consecutive_core_errors
        self._on_frame_processed = on_frame_processed
        self._consecutive_core_errors = 0

    def process_frame(self, frame: Frame) -> FrameContext:
        """Run one already-captured frame through the full pipeline.

        Returns the `FrameContext` describing what happened -- always,
        whether the core chain succeeded or a stage in it raised. A
        core-chain exception is caught here, recorded in
        `context.errors`, and never propagates: every stage's persistent
        state is left exactly as it was before this call (see module
        docstring). Export/debug only run for a frame whose core chain
        succeeded.
        """
        context = FrameContext(
            frame_id=frame.frame_index, timestamp=frame.timestamp, raw_frame=frame
        )
        try:
            context.detections = self._detector.detect(frame)

            tracker_update = self._tracker.compute_update(context.detections)
            hysteresis_update = self._hysteresis.compute_update(tracker_update.tracked_frame)
            context.tracks = hysteresis_update.tracked_frame

            assignments = assign_zones(context.tracks, self._calibration)
            context.assignments = apply_dealer_nearest_seat_fallback(
                context.tracks,
                assignments,
                self._calibration,
                self._dealer_nearest_seat_max_distance,
            )

            state_update = self._state_machine.compute_update(context.assignments)
        except Exception as exc:
            self._consecutive_core_errors += 1
            context.errors.append(f"{type(exc).__name__}: {exc}")
            logger.error(
                "frame %d: core chain failed (%d consecutive failure(s)): %s",
                frame.frame_index,
                self._consecutive_core_errors,
                exc,
                exc_info=True,
            )
            self._notify(context)
            return context

        # The entire core chain succeeded for this frame -- only now do we
        # commit each stage's pure update, in pipeline order.
        self._tracker.commit(tracker_update)
        self._hysteresis.commit(hysteresis_update)
        context.events = self._state_machine.commit(state_update)
        context.state_snapshot = self._state_machine.snapshot()
        self._consecutive_core_errors = 0

        # export: ExportManager isolates every adapter's own failures and
        # never raises itself (REQ-37a) -- nothing to catch here.
        self._export_manager.export(context.events)

        if self._debug_server is not None:
            try:
                self._debug_server.update_frame(frame, context.tracks, context.assignments)
            except Exception:
                logger.exception("frame %d: debug overlay update failed", frame.frame_index)

        self._notify(context)
        return context

    def run(self) -> LoopExitReason:
        """Process frames from `capture` until EOF or a fatal error.

        Returns `LoopExitReason.EOF` once the capture source is exhausted
        (`video_file`/`image_dir` raising `StopIteration` -- REQ-44's "EOF
        ... beendet den Loop regulär"; `continuity` never does, see
        REQ-16). Raises `FatalPipelineError` once
        `max_consecutive_core_errors` core-chain frames in a row have
        failed. Any other exception -- notably a live `continuity` read
        failure -- propagates as-is.
        """
        for frame in self._capture:
            self.process_frame(frame)
            if self._consecutive_core_errors >= self._max_consecutive_core_errors:
                raise FatalPipelineError(
                    f"{self._consecutive_core_errors} consecutive core-chain failures "
                    f"(max {self._max_consecutive_core_errors}); aborting"
                )
        return LoopExitReason.EOF

    def _notify(self, context: FrameContext) -> None:
        if self._on_frame_processed is not None:
            self._on_frame_processed(context)
