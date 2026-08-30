"""MJPEG debug endpoint with live overlay (REQ-37), plus the static debug
page that combines it with the WebSocket event list (REQ-38).

Serves the same kind of "what does the pipeline currently see" view Phase 0
produced as a single static image (zones, track IDs, rubber-band
track -> seat line with distance), but as a live `multipart/x-mixed-replace`
stream over FastAPI/uvicorn, plus the occupancy/dealer/street state from
`PipelineStateMachine.snapshot()` that Phase 0 had no equivalent of. Actual
drawing lives in `debug.overlay` -- this module is only the pipeline-facing
`update_frame()` call and the ASGI plumbing around it, the same split
`export.websocket` draws between event serialization and its FastAPI app.

`build_debug_server()` is the config-driven on/off switch (REQ-37, "über
Config abschaltbar"): when `Config.debug.enabled` is `False`, it returns
`None` and no `MjpegDebugServer` -- hence no FastAPI route, no uvicorn
server -- is ever constructed, mirroring `export.manager.build_exporters()`'s
per-adapter enable pattern (REQ-37a).

`update_frame()` is the same synchronous, fire-and-forget call the pipeline
already makes against every export adapter's `export()`: it renders the
overlay and stores the encoded JPEG bytes under a lock, from whatever
thread the pipeline's own frame loop runs on. Each `/mjpeg` connection's
async generator polls that stored buffer on its own ASGI event loop and
streams a new multipart part whenever it changes -- the same lock-protected
shared-state bridge `WebSocketEventExporter` (REQ-35) uses between the
pipeline thread and uvicorn's loop, but adapted from a per-connection queue
(every distinct message must be delivered) to a single "latest frame" slot
(only the newest frame is worth showing; a slow consumer skips stale ones
rather than backing up).

`GET /` serves `static/index.html` (REQ-38): a plain HTML/CSS/JS file, no
frontend framework and no build step, that shows the `/mjpeg` stream next
to a live list fed by the `websocket` export adapter's `/ws` (REQ-35). The
only thing not fixed at authoring time is which port that adapter runs
on, since it's a separate FastAPI app from this one -- `_render_debug_page()`
fills that one placeholder into the static file's text once at server
construction, which is a fixed startup-time substitution, not a build step
or per-request templating.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from poker_vision.assignment.models import FrameAssignments
from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.capture.frame import Frame
from poker_vision.config import Config, PortsConfig
from poker_vision.debug.overlay import render_overlay
from poker_vision.state.machine import PipelineStateMachine
from poker_vision.tracking.models import TrackedFrame

_BOUNDARY = b"frame"
_POLL_INTERVAL_SECONDS = 0.01
_JPEG_ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
_DEBUG_PAGE_TEMPLATE = Path(__file__).parent / "static" / "index.html"


class MjpegDebugServer:
    """Renders and serves the debug overlay as an MJPEG stream (REQ-37),
    plus the static debug page that combines it with the WebSocket event
    list (REQ-38).
    """

    def __init__(
        self,
        calibration: CalibrationRuntime,
        state_machine: PipelineStateMachine,
        websocket_port: int = PortsConfig().websocket,
    ) -> None:
        self._calibration = calibration
        self._state_machine = state_machine
        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        # `websocket_port` is baked into the page once here rather than on
        # every request: the value is fixed for this server's lifetime (it
        # comes from `Config.ports.websocket`), so re-rendering it per
        # request would just repeat the same substitution for no benefit.
        # The page itself stays a genuinely static HTML/CSS/JS file on disk
        # (REQ-38, "ohne Frontend-Framework und ohne Build-Schritt") -- this
        # is a single startup-time placeholder substitution, not templating
        # per request and not a build step.
        self._debug_page = _render_debug_page(websocket_port)
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/")
        def debug_page() -> HTMLResponse:
            return HTMLResponse(self._debug_page)

        @app.get("/mjpeg")
        def mjpeg(request: Request) -> StreamingResponse:
            return StreamingResponse(
                self._stream(request),
                media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}",
            )

    def update_frame(
        self,
        frame: Frame,
        tracked_frame: TrackedFrame,
        frame_assignments: FrameAssignments,
    ) -> None:
        """Render this frame's overlay and store it as the latest JPEG to stream.

        Called once per processed frame from the pipeline's own frame loop,
        the same way `PipelineStateMachine.update()` and every export
        adapter's `export()` are.
        """
        annotated = render_overlay(
            frame.image,
            self._calibration,
            tracked_frame,
            frame_assignments,
            self._state_machine.snapshot(),
        )
        jpeg_bytes = _encode_jpeg(annotated)
        with self._lock:
            self._latest_jpeg = jpeg_bytes

    async def _stream(self, request: Request):
        # `StreamingResponse` never stops this generator on its own -- unlike
        # `WebSocketEventExporter` (REQ-35), which notices a disconnect via
        # its own `receive()` poll, a plain HTTP streaming response has no
        # equivalent built-in signal. Without polling `request.is_
        # disconnected()`, a client that goes away (or a test that opens and
        # closes the connection without reading it to completion) would
        # leave this coroutine looping forever.
        last_sent: bytes | None = None
        while not await request.is_disconnected():
            with self._lock:
                jpeg_bytes = self._latest_jpeg
            if jpeg_bytes is not None and jpeg_bytes is not last_sent:
                yield (
                    b"--" + _BOUNDARY + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg_bytes)).encode() + b"\r\n\r\n"
                    + jpeg_bytes
                    + b"\r\n"
                )
                last_sent = jpeg_bytes
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    def run(self, host: str = "0.0.0.0", port: int = 8001) -> None:
        """Block, serving this server's FastAPI app via uvicorn."""
        uvicorn.run(self.app, host=host, port=port)


def _encode_jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, _JPEG_ENCODE_PARAMS)
    if not ok:
        raise ValueError("failed to JPEG-encode the debug overlay frame")
    return encoded.tobytes()


def _render_debug_page(websocket_port: int) -> str:
    """Fill the one placeholder `static/index.html` carries: the WebSocket
    port to connect to. The page loads `/mjpeg` as a same-origin relative
    URL (it's served by this same `MjpegDebugServer`), but the WebSocket
    export adapter (REQ-35) runs as a separate FastAPI app on its own port,
    so the page has no other way to learn it without going back to the
    server for it -- there is no per-connection origin to infer it from.
    """
    template = _DEBUG_PAGE_TEMPLATE.read_text(encoding="utf-8")
    return template.replace("{{WS_PORT}}", str(websocket_port))


def build_debug_server(
    config: Config, calibration: CalibrationRuntime, state_machine: PipelineStateMachine
) -> MjpegDebugServer | None:
    """Construct the MJPEG debug server iff `config.debug.enabled` (REQ-37, AC-24).

    Returns `None` when disabled, so the caller never constructs a
    `FastAPI` app or starts a uvicorn server for it -- "debug.enabled: false
    startet keinen MJPEG-Endpoint" (AC-24), not merely one that draws
    nothing.
    """
    if not config.debug.enabled:
        return None
    return MjpegDebugServer(calibration, state_machine, websocket_port=config.ports.websocket)
