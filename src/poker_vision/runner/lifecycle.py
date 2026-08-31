"""CLI-facing lifecycle: config-driven pipeline construction, SIGINT/SIGTERM
shutdown and exit codes (REQ-45).

Owns everything `runner/loop.py`'s `FrameLoop` deliberately doesn't
(see that module's docstring): loading + validating `Config` and
`CalibrationRuntime`, constructing every pipeline stage from them,
starting/stopping the debug server, and mapping every abort condition to
an exit code. `runner/cli.py` is a thin argument-parsing wrapper around
`run_command()`/`validate_command()` below.

## Exit codes

- `0` (`EXIT_OK`): the pipeline ran to completion -- EOF on a
  `video_file`/`image_dir` source, or a graceful SIGINT/SIGTERM shutdown.
- `2` (`EXIT_CONFIG_ERROR`): the config file is missing/malformed/invalid,
  including a `mock` detector with an ambiguous or empty mode selection
  (`detection.create_detector`) -- content the schema itself can't check
  eagerly but that's still fundamentally a config problem, not a runtime
  one.
- `3` (`EXIT_CALIBRATION_ERROR`): the calibration file is
  missing/malformed/invalid (REQ-11's zone/topology checks included --
  they run as part of loading it).
- `4` (`EXIT_PIPELINE_ERROR`): an error-threshold abort once the loop was
  already under way or about to start -- `max_consecutive_core_errors`
  core-chain failures in a row (`FatalPipelineError`), the capture
  couldn't be opened at all on the very first attempt, or (`continuity`
  only) `source.continuity_retry.timeout_seconds` of continuous capture
  failure (`ContinuityRetryExhausted`).
- `130` (`EXIT_FORCED_ABORT`): a second SIGINT forced an immediate,
  unclean exit (`os._exit`) -- the conventional "killed by SIGINT"
  status (128 + signal number 2), distinguishing it from every graceful
  path above.
- `1` (`EXIT_UNEXPECTED_ERROR`): anything else -- see `cli.main()`'s own
  catch-all.

## Shutdown sequence

Mirrors CLAUDE.md's "Lifecycle" section exactly: capture close -> export
flush/close -> debug server stop -> process exit, run from a single
`finally` so it happens the same way whether the loop stopped at EOF, on
a shutdown request, or because an exception aborted it.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import FrameType

import uvicorn
from fastapi import FastAPI

from poker_vision.calibration.runtime import CalibrationRuntime, load_calibration_runtime
from poker_vision.capture import create_capture
from poker_vision.capture.base import Capture
from poker_vision.capture.frame import Frame
from poker_vision.config import Config, ContinuityRetryConfig, SourceType, load_config
from poker_vision.debug.mjpeg import MjpegDebugServer, build_debug_server
from poker_vision.detection.base import Detector
from poker_vision.detection.factory import create_detector
from poker_vision.export.manager import ExportManager, build_exporters
from poker_vision.export.websocket import WebSocketEventExporter
from poker_vision.runner.context import FrameContext
from poker_vision.runner.loop import FatalPipelineError, FrameLoop, LoopExitReason
from poker_vision.state.machine import PipelineStateMachine
from poker_vision.tracking import create_tracker
from poker_vision.tracking.hysteresis import HysteresisFilter
from poker_vision.tracking.tracker import NearestMatchTracker

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_UNEXPECTED_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_CALIBRATION_ERROR = 3
EXIT_PIPELINE_ERROR = 4
EXIT_FORCED_ABORT = 128 + signal.SIGINT  # 130, the conventional "killed by SIGINT" status


class ContinuityRetryExhausted(RuntimeError):
    """A `continuity` capture kept failing for `timeout_seconds`; giving up.

    Distinct from `FatalPipelineError` (too many core-chain failures) and
    from a plain capture-open `RuntimeError` (the very first attempt,
    never retried -- see `_run_capture_with_retry`'s docstring) so
    `run_command` can log a message specific to each, even though all
    three map to the same exit code.
    """


class ShutdownController:
    """Installs SIGINT/SIGTERM handlers for a graceful, signal-driven stop.

    First SIGINT or SIGTERM sets a flag (`requested()`) that `FrameLoop.
    run()` checks once after each frame finishes -- "aktueller Frame wird
    abgeschlossen" -- and that `_run_capture_with_retry`'s backoff sleep
    (`wait()`) also wakes up on early, so a shutdown mid-retry-backoff
    doesn't sit out the rest of it. A second SIGINT is REQ-45's own
    "Zweites SIGINT erzwingt sofortigen Abbruch": since the graceful path
    can be stuck on a slow/hung capture read that never reaches the
    per-frame check above, the only way to *guarantee* an immediate stop
    is `os._exit()` from directly inside the handler.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._sigint_count = 0
        self._previous_handlers: dict[
            int, Callable[[int, FrameType | None], object] | int | None
        ] = {}

    def install(self) -> None:
        self._previous_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._handle)
        self._previous_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._handle)

    def restore(self) -> None:
        for sig, handler in self._previous_handlers.items():
            signal.signal(sig, handler)
        self._previous_handlers.clear()

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        if signum == signal.SIGINT:
            self._sigint_count += 1
            if self._sigint_count >= 2:
                logger.warning("second SIGINT received; forcing an immediate, unclean exit")
                os._exit(EXIT_FORCED_ABORT)
        logger.info(
            "received signal %s; stopping after the current frame finishes",
            signal.Signals(signum).name,
        )
        self._event.set()

    def requested(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> None:
        """Sleep up to `timeout` seconds, waking early if shutdown is requested."""
        self._event.wait(timeout)

    def wait_forever(self) -> None:
        """Block until shutdown is requested -- never times out."""
        self._event.wait()


class _CaptureInterruptWatcher:
    """Makes a blocked `continuity` read respond to the *first*
    SIGINT/SIGTERM instead of only the second (Codex review, REQ-45).

    `should_stop` (checked between frames) can't help when `capture.
    __next__()` itself never returns -- a live camera that's still open
    but has stopped delivering frames blocks inside `ContinuityCapture.
    __next__()`'s own internal wait loop indefinitely. That class's
    `close()` is explicitly documented to "unblock any waiting __next__()/
    get_latest() call" from another thread, which is exactly what's
    needed here: a small daemon thread that does nothing until shutdown
    is requested, then closes whichever capture is current at that moment
    -- from a thread other than the (possibly still blocked) main one.
    `_run_capture_with_retry` re-registers the current capture on every
    reconnect via `set_current()`, and must route its own per-attempt
    cleanup through `close_current()` too (not call `capture.close()`
    directly) -- `Capture.close()`'s "safe to call more than once"
    contract covers repeated *sequential* calls, not necessarily two
    threads calling it at the exact same instant (e.g. `ContinuityCapture.
    close()` isn't lock-guarded, so concurrent calls could race inside the
    underlying `cv2.VideoCapture.release()`). `close_current()` swaps
    `_current` to `None` under its own lock before closing, so whichever
    of the watcher thread or the main retry loop gets there first is the
    only one that actually calls `capture.close()`.

    The watch loop keeps re-checking after shutdown is first observed,
    rather than firing once and exiting (Codex review): shutdown could be
    requested in the narrow window between `create_capture()` returning
    and `set_current()` registering it (e.g. a slow-to-open device), in
    which case a one-shot watcher would already be gone and this new
    capture would never get interrupted if it then blocked. Once shutdown
    is set, `ShutdownController.wait_forever()` returns immediately on
    every call, so the loop below polls `close_current()` on a short
    interval instead of busy-spinning.
    """

    _REARM_POLL_INTERVAL_SECONDS = 0.05

    def __init__(self, shutdown: ShutdownController) -> None:
        self._shutdown = shutdown
        self._lock = threading.Lock()
        self._current: Capture | None = None
        self._thread = threading.Thread(
            target=self._watch, name="capture-interrupt-watcher", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def set_current(self, capture: Capture | None) -> None:
        with self._lock:
            self._current = capture

    def close_current(self) -> None:
        with self._lock:
            capture, self._current = self._current, None
        if capture is not None:
            capture.close()

    def _watch(self) -> None:
        self._shutdown.wait_forever()
        while True:
            self.close_current()
            time.sleep(self._REARM_POLL_INTERVAL_SECONDS)


@dataclass
class _ServerHandle:
    """A `uvicorn` app running on its own background thread (daemon, so a
    forced-abort exit never waits on it), stoppable without killing the
    whole process.
    """

    server: uvicorn.Server
    thread: threading.Thread

    def stop(self, timeout: float = 5.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=timeout)


def _run_uvicorn_quietly(server: uvicorn.Server) -> None:
    """`server.run()`, swallowing the `SystemExit` uvicorn itself raises on
    a startup failure (e.g. the port is already in use) -- the caller
    already detects and reports that failure via `_start_uvicorn_background`
    checking `server.started`; without this, the same failure would *also*
    surface as a scary, redundant "unhandled exception in thread"
    traceback on stderr.
    """
    try:
        server.run()
    except SystemExit:
        pass


def _start_uvicorn_background(app: FastAPI, host: str, port: int) -> _ServerHandle:
    """Start `app` on its own background uvicorn thread, blocking until it
    has actually bound its socket (or failed to).

    Raises `RuntimeError` if the server isn't listening within the
    deadline -- e.g. `port` already in use (uvicorn's own thread exits
    early) or startup simply hanging (Codex review: silently returning a
    handle either way would let a caller believe an endpoint is open when
    it isn't).
    """
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(
        target=_run_uvicorn_quietly, args=(server,), name=f"uvicorn-{port}", daemon=True
    )
    thread.start()
    # Block briefly until the server has actually bound its socket, so a
    # caller that immediately starts driving the pipeline doesn't race a
    # client connecting before the port is open.
    deadline = time.monotonic() + 5.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        thread.join(timeout=5.0)
        raise RuntimeError(f"server on {host}:{port} did not start (port in use, or timed out)")
    return _ServerHandle(server, thread)


class _CaptureHealthTracker:
    """Tracks whether the current `capture` attempt has delivered at least
    one frame -- `_run_capture_with_retry`'s signal that a reconnect
    actually succeeded (resetting the failure-timeout window) -- and, if
    so, the highest (already globally-numbered, see `_OffsetCapture`)
    `frame_index` it delivered, so the next reconnect attempt can keep
    numbering frames from there.
    """

    def __init__(self) -> None:
        self.frame_seen = False
        self.last_frame_index: int | None = None

    def mark(self, context: FrameContext) -> None:
        self.frame_seen = True
        self.last_frame_index = context.frame_id


@dataclass
class _OffsetCapture:
    """Wraps a freshly (re)opened `Capture`, adding a fixed `offset` to
    every yielded frame's `frame_index` -- see `_run_capture_with_retry`'s
    docstring for why this is necessary across a `continuity` reconnect.
    """

    inner: Capture
    offset: int

    @property
    def source_id(self) -> str:
        return self.inner.source_id

    def __iter__(self) -> _OffsetCapture:
        return self

    def __next__(self) -> Frame:
        frame = next(self.inner)
        if self.offset == 0:
            return frame
        return replace(frame, frame_index=frame.frame_index + self.offset)

    def close(self) -> None:
        self.inner.close()


@dataclass
class _RetryWindow:
    """Tracks how long continuity capture failures have persisted
    *continuously* (REQ-45: bound one outage, not cumulative downtime --
    see `ContinuityRetryConfig`'s docstring).

    `record_failure()` and `is_exhausted()` are separate (not one
    combined call) so `_handle_continuity_failure` can check exhaustion
    both *before* and *after* its backoff sleep against the same
    `started_at` -- otherwise a `backoff_seconds` close to or larger than
    `timeout_seconds` could push the total outage past the timeout
    without either check ever actually seeing it (Codex review): the
    pre-sleep check always fires too early (elapsed is still ~0 right
    after the first failure) to catch it, and if the reconnect
    immediately following the sleep happens to succeed, no *further*
    failure ever triggers a fresh check either.
    """

    started_at: float | None = None

    def reset(self) -> None:
        self.started_at = None

    def record_failure(self) -> None:
        if self.started_at is None:
            self.started_at = time.monotonic()

    def is_exhausted(self, timeout_seconds: float) -> bool:
        if self.started_at is None:
            return False
        return (time.monotonic() - self.started_at) >= timeout_seconds


def _run_capture_with_retry(
    config: Config,
    shutdown: ShutdownController,
    calibration: CalibrationRuntime,
    detector: Detector,
    tracker: NearestMatchTracker,
    hysteresis: HysteresisFilter,
    state_machine: PipelineStateMachine,
    export_manager: ExportManager,
    debug_server: MjpegDebugServer | None,
) -> LoopExitReason:
    """Run the frame loop, reopening a failed `continuity` capture with backoff.

    Every stage passed in here is long-lived across reconnects -- in
    particular `tracker`/`hysteresis`/`state_machine` keep their in-memory
    state (track IDs, hysteresis counts, hand/street state) exactly as it
    was when the capture failed, so a camera hiccup never resets an
    in-progress hand. Only `capture` (and the `FrameLoop` wrapping it) is
    rebuilt on each attempt.

    The *very first* `create_capture()` call is never retried, regardless
    of `source.type`: a `continuity` camera missing before the pipeline
    has ever run at all is REQ-16's "klarer Fehler, kein Fallback", not an
    outage to ride out. Once that first attempt has succeeded, every
    subsequent failure -- a live read failing, or the reconnect itself
    failing -- is retried for `continuity` sources only, with backoff,
    until `source.continuity_retry.timeout_seconds` of *continuous*
    failure is reached (`ContinuityRetryExhausted`) or shutdown is
    requested -- in which case this returns `LoopExitReason.
    SHUTDOWN_REQUESTED` (a graceful stop, exit 0), not
    `ContinuityRetryExhausted` (exit != 0): a SIGINT/SIGTERM landing
    during a retry backoff is still a graceful shutdown request, same as
    one landing between two frames. `video_file`/`image_dir` never retry
    at all (REQ-15).

    Every reconnected `ContinuityCapture` restarts its own `frame_index`
    at 0 (see its own module docstring), but `HysteresisFilter`/
    `PipelineStateMachine` -- kept alive across reconnects precisely so an
    in-progress hand survives a camera hiccup -- both require a strictly
    increasing `frame_index` from call to call. `_OffsetCapture` keeps the
    numbering continuous across reconnects by adding a running offset
    (the last globally-numbered frame index actually delivered, plus one)
    to every frame from a freshly (re)opened capture.

    A `continuity` capture that's still open but has stopped delivering
    frames would otherwise block inside `capture.__next__()` past the
    first SIGINT/SIGTERM -- `should_stop` (checked between frames, not
    inside a blocked read) can't help there. `_CaptureInterruptWatcher`
    closes the current capture as soon as shutdown is requested, from a
    separate thread, so that blocked read is interrupted promptly instead
    of only on a forced second SIGINT (`ContinuityCapture.close()`'s own
    documented contract).
    """
    is_continuity = config.source.type is SourceType.CONTINUITY
    retry_config = config.source.continuity_retry
    window = _RetryWindow()
    opened_before = False
    next_frame_index_offset = 0
    watcher: _CaptureInterruptWatcher | None = None
    if is_continuity:
        watcher = _CaptureInterruptWatcher(shutdown)
        watcher.start()

    while True:
        try:
            raw_capture: Capture = create_capture(config.source)
        except RuntimeError as exc:
            if not is_continuity or not opened_before:
                raise
            if _handle_continuity_failure(exc, window, retry_config, shutdown, context="reopen"):
                continue
            if shutdown.requested():
                return LoopExitReason.SHUTDOWN_REQUESTED
            raise ContinuityRetryExhausted(
                f"continuity capture failed to reopen for >= {retry_config.timeout_seconds}s: {exc}"
            ) from exc
        opened_before = True
        capture: Capture = (
            _OffsetCapture(raw_capture, next_frame_index_offset) if is_continuity else raw_capture
        )
        if watcher is not None:
            watcher.set_current(capture)

        health = _CaptureHealthTracker()
        loop = FrameLoop(
            capture=capture,
            detector=detector,
            tracker=tracker,
            hysteresis=hysteresis,
            calibration=calibration,
            dealer_nearest_seat_max_distance=config.thresholds.dealer_nearest_seat_max_distance,
            state_machine=state_machine,
            export_manager=export_manager,
            debug_server=debug_server,
            max_consecutive_core_errors=config.runner.max_consecutive_core_errors,
            on_frame_processed=health.mark,
        )
        try:
            return loop.run(should_stop=shutdown.requested)
        except FatalPipelineError:
            raise
        except Exception as exc:
            # `ContinuityCapture`'s background reader thread republishes
            # *any* exception from `cv2.VideoCapture.read()` or frame
            # construction (see its module docstring's `except Exception`
            # note), not only `RuntimeError` -- a bare `except RuntimeError`
            # here would let e.g. a `cv2.error` bypass the retry policy
            # entirely and get reported as an immediate pipeline failure
            # (Codex review). `FatalPipelineError` is excluded above, and
            # `not is_continuity` still re-raises immediately: `video_file`/
            # `image_dir` never retry (REQ-15).
            if not is_continuity:
                raise
            if health.frame_seen:
                window.reset()
                next_frame_index_offset = health.last_frame_index + 1
            if _handle_continuity_failure(exc, window, retry_config, shutdown, context="read"):
                continue
            if shutdown.requested():
                return LoopExitReason.SHUTDOWN_REQUESTED
            raise ContinuityRetryExhausted(
                f"continuity capture kept failing for >= {retry_config.timeout_seconds}s: {exc}"
            ) from exc
        finally:
            if watcher is not None:
                watcher.close_current()
            else:
                capture.close()


def _handle_continuity_failure(
    exc: Exception,
    window: _RetryWindow,
    retry_config: ContinuityRetryConfig,
    shutdown: ShutdownController,
    context: str,
) -> bool:
    """Record one continuity failure; return whether to retry (vs. give up).

    Also gives up immediately -- without waiting out the rest of the
    backoff -- once shutdown has been requested, so a SIGINT/SIGTERM
    during a retry backoff doesn't delay the process exit.
    """
    window.record_failure()
    if window.is_exhausted(retry_config.timeout_seconds) or shutdown.requested():
        return False
    logger.warning(
        "continuity capture %s failed (%s); retrying in %.1fs",
        context,
        exc,
        retry_config.backoff_seconds,
    )
    shutdown.wait(retry_config.backoff_seconds)
    if shutdown.requested():
        return False
    # Recheck (Codex review): the backoff sleep itself can push the total
    # outage past the timeout even when the pre-sleep check hadn't yet --
    # see `_RetryWindow`'s own docstring for why this can't be caught any
    # other way.
    return not window.is_exhausted(retry_config.timeout_seconds)


_Stages = tuple[
    Detector,
    NearestMatchTracker,
    HysteresisFilter,
    PipelineStateMachine,
    ExportManager,
    MjpegDebugServer | None,
]


def _build_stages(config: Config, calibration: CalibrationRuntime) -> _Stages:
    """Construct every long-lived stage (everything except `capture`,
    rebuilt per reconnect -- see `_run_capture_with_retry`).

    Raises `ValueError` on an unbuildable config (currently: an
    ambiguous/empty mock detector mode -- `detection.create_detector`),
    mapped by `run_command` to `EXIT_CONFIG_ERROR` the same as a `load_
    config()` failure, since it's the same kind of problem, just caught
    later.
    """
    detector = create_detector(config, calibration)
    tracker = create_tracker(config, calibration.table)
    hysteresis = HysteresisFilter(config.hysteresis)
    state_machine = PipelineStateMachine(seat.seat_id for seat in calibration.seats)
    export_manager = ExportManager(build_exporters(config, state_machine))
    debug_server = build_debug_server(config, calibration, state_machine)
    return detector, tracker, hysteresis, state_machine, export_manager, debug_server


def validate_command(config_path: str | Path) -> int:
    """`poker-vision validate --config <path>`: check config + calibration,
    including REQ-11's zone/topology validation, without starting the loop.
    """
    try:
        config = load_config(config_path)
    except ValueError as exc:
        logger.error("config invalid: %s", exc)
        return EXIT_CONFIG_ERROR

    try:
        calibration = load_calibration_runtime(config.paths.calibration_runtime)
    except ValueError as exc:
        logger.error("calibration invalid: %s", exc)
        return EXIT_CALIBRATION_ERROR

    try:
        # Only the detector, not the full `_build_stages()`: constructing
        # the export/debug stages too would have real side effects (e.g.
        # `JsonlEventExporter` creates its export directory and opens a
        # session file) that a mere validation pass shouldn't cause.
        # `create_detector` is the one piece of "Stufen konstruieren" that
        # can reject an otherwise schema-valid config (an ambiguous/empty
        # mock-mode selection -- see its own docstring), so `validate`
        # must check it too, or a config `run` would reject as invalid
        # could still pass `validate` (Codex review finding).
        create_detector(config, calibration)
    except (ValueError, OSError) as exc:
        logger.error("config invalid: %s", exc)
        return EXIT_CONFIG_ERROR

    logger.info("config and calibration are valid")
    return EXIT_OK


def run_command(config_path: str | Path) -> int:
    """`poker-vision run --config <path>`: start the pipeline and run it
    until EOF or a signal-driven shutdown, per this module's docstring.
    """
    try:
        config = load_config(config_path)
    except ValueError as exc:
        logger.error("config invalid: %s", exc)
        return EXIT_CONFIG_ERROR

    try:
        calibration = load_calibration_runtime(config.paths.calibration_runtime)
    except ValueError as exc:
        logger.error("calibration invalid: %s", exc)
        return EXIT_CALIBRATION_ERROR

    try:
        detector, tracker, hysteresis, state_machine, export_manager, debug_server = (
            _build_stages(config, calibration)
        )
    except (ValueError, OSError) as exc:
        # `OSError` included: e.g. Modus A's `MockDetector` raises it for a
        # missing/unreadable `paths.mock_script` -- the same kind of
        # "invalid config" `validate_command` already classifies this way
        # via the same `create_detector()` call (Codex review).
        logger.error("config invalid: %s", exc)
        return EXIT_CONFIG_ERROR

    debug_handle: _ServerHandle | None = None
    if debug_server is not None:
        try:
            debug_handle = _start_uvicorn_background(
                debug_server.app, "0.0.0.0", config.ports.mjpeg
            )
        except RuntimeError as exc:
            # `debug` is best-effort (CLAUDE.md's Fehlerpolitik: a
            # rendering/publish failure never stops the loop) -- extended
            # here to a startup failure too (e.g. `ports.mjpeg` already in
            # use): log it loudly (Codex review: it must not be silent)
            # and run without it rather than aborting an otherwise-healthy
            # pipeline over a debugging convenience endpoint.
            logger.error(
                "debug server failed to start on port %d (%s); running without it",
                config.ports.mjpeg,
                exc,
            )
            debug_server = None

    # `build_exporters()` (REQ-37a) constructs a `WebSocketEventExporter`
    # when `export.websocket` is enabled, but constructing it doesn't
    # serve its FastAPI app anywhere -- that's this lifecycle's job,
    # exactly like the debug server above (Codex review: without this,
    # nothing ever listens on `ports.websocket`, so REQ-35's /ws, /status,
    # /health are all unreachable despite the adapter being "enabled").
    websocket_exporter = next(
        (e for e in export_manager.exporters if isinstance(e, WebSocketEventExporter)), None
    )
    websocket_handle: _ServerHandle | None = None
    if websocket_exporter is not None:
        try:
            websocket_handle = _start_uvicorn_background(
                websocket_exporter.app, "0.0.0.0", config.ports.websocket
            )
        except RuntimeError as exc:
            # Same best-effort philosophy as every other export adapter
            # (REQ-37a: "Ausfall eines Adapters stoppt die Pipeline
            # nicht"), extended to this adapter's startup phase too.
            logger.error(
                "websocket export server failed to start on port %d (%s); running without it",
                config.ports.websocket,
                exc,
            )

    shutdown = ShutdownController()
    shutdown.install()
    try:
        reason = _run_capture_with_retry(
            config,
            shutdown,
            calibration,
            detector,
            tracker,
            hysteresis,
            state_machine,
            export_manager,
            debug_server,
        )
        logger.info("pipeline stopped: %s", reason.value)
        return EXIT_OK
    except (FatalPipelineError, ContinuityRetryExhausted) as exc:
        logger.error("pipeline aborted: %s", exc)
        return EXIT_PIPELINE_ERROR
    except (RuntimeError, OSError, ValueError) as exc:
        # The very first capture-open failure (never retried -- see
        # `_run_capture_with_retry`'s docstring): `RuntimeError` for
        # `continuity` (REQ-16's "camera missing at startup"),
        # `FileNotFoundError`/`ValueError` for `video_file`/`image_dir`
        # (a missing file, an empty/unreadable directory).
        logger.error("pipeline failed to start: %s", exc)
        return EXIT_PIPELINE_ERROR
    finally:
        shutdown.restore()
        export_manager.close()
        if debug_handle is not None:
            debug_handle.stop()
        if websocket_handle is not None:
            websocket_handle.stop()
