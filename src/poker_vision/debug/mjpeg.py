"""MJPEG debug endpoint with live overlay (REQ-37), plus the static debug
page that combines it with the WebSocket event list (REQ-38).

Serves the same kind of "what does the pipeline currently see" view Phase 0
produced as a single static image (zones, track IDs, rubber-band
track -> seat line with distance), but as a live `multipart/x-mixed-replace`
stream over FastAPI/uvicorn, plus the occupancy/dealer/street state that
Phase 0 had no equivalent of. Actual drawing lives in `debug.overlay`; the
pipeline-facing side of this module isn't `MjpegDebugServer` at all but
`debug.frame_hub.LatestFrameHub`, which the loop (`runner/loop.py`)
publishes into directly, once per successfully processed frame (REQ-46) --
this module only ever *reads* that same hub, from `_stream()`.

`build_debug_server()` is the config-driven on/off switch (REQ-37, "über
Config abschaltbar"): when `Config.debug.enabled` is `False`, it returns
`None` and no `MjpegDebugServer` -- hence no FastAPI route, no uvicorn
server -- is ever constructed, mirroring `export.manager.build_exporters()`'s
per-adapter enable pattern (REQ-37a).

REQ-46 moved overlay rendering from eager (every `publish()` call, whether
or not anyone was watching) to on-demand: `_stream()` is the *only* place
`render_overlay()` is called from, and it only runs while a client's
`/mjpeg` connection is open and being iterated -- "ohne verbundenen Client
findet kein Rendering statt". Each connected client's async generator
blocks (via `asyncio.to_thread`, so it never stalls the shared ASGI event
loop or any other connected client) on `LatestFrameHub.get_latest()` for a
version newer than the last one it rendered, renders + JPEG-encodes that
frame itself, and streams the result -- independently of every other
client and of the loop, which never waits on any of this (`LatestFrameHub.
publish()` never renders or encodes anything).

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
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from poker_vision.calibration.runtime import CalibrationRuntime
from poker_vision.config import Config, PortsConfig
from poker_vision.debug.frame_hub import DebugSnapshot, LatestFrameHub
from poker_vision.debug.overlay import render_overlay

_BOUNDARY = b"frame"
# How long each client's `_stream()` blocks per `get_latest()` call before
# re-checking `request.is_disconnected()` -- a "kurzer blockierender Wait
# mit Timeout" (CLAUDE.md), not a value observed elsewhere: a timeout just
# means "no newer frame yet, keep waiting", trading off disconnect
# responsiveness against wakeup frequency the same way `capture.continuity`'s
# `_DEFAULT_WAIT_TIMEOUT_SECONDS` does for the opposite direction.
_WAIT_TIMEOUT_SECONDS = 0.5
_JPEG_ENCODE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
_DEBUG_PAGE_TEMPLATE = Path(__file__).parent / "static" / "index.html"


class MjpegDebugServer:
    """Serves the debug overlay as an MJPEG stream, rendered on demand per
    connected client from a shared `LatestFrameHub` (REQ-37, REQ-46), plus
    the static debug page that combines it with the WebSocket event list
    (REQ-38).
    """

    def __init__(
        self,
        calibration: CalibrationRuntime,
        frame_hub: LatestFrameHub,
        websocket_port: int = PortsConfig().websocket,
    ) -> None:
        self._calibration = calibration
        self.frame_hub = frame_hub
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

    async def _stream(self, request: Request):
        # `StreamingResponse` never stops this generator on its own -- unlike
        # `WebSocketEventExporter` (REQ-35), which notices a disconnect via
        # its own `receive()` poll, a plain HTTP streaming response has no
        # equivalent built-in signal. Without polling `request.is_
        # disconnected()`, a client that goes away (or a test that opens and
        # closes the connection without reading it to completion) would
        # leave this coroutine looping forever.
        #
        # `since_version` is this one client's own read position into the
        # shared hub (REQ-46: `LatestFrameHub` supports any number of
        # independent readers, unlike `capture.continuity`'s single-consumer
        # buffer) -- starts at 0 so the very first iteration renders
        # whatever is already published, if anything.
        since_version = 0
        while not await request.is_disconnected():
            # `get_latest()` blocks (with a timeout) inside `threading.
            # Condition.wait_for` -- run it off the event loop via
            # `asyncio.to_thread` so it can never stall this or any other
            # client's coroutine sharing the same uvicorn event loop.
            result = await asyncio.to_thread(
                self.frame_hub.get_latest, since_version, _WAIT_TIMEOUT_SECONDS
            )
            if result is None:
                continue
            frame, snapshot, since_version = result
            # On-demand rendering (REQ-46): this is the only call to
            # `render_overlay()` in the whole server, and it only runs here
            # -- while a client is connected and this generator is being
            # iterated -- so a `publish()` with no client watching never
            # renders or encodes anything. Also run off the event loop
            # (Codex review): `render_overlay`/JPEG encoding are CPU-bound
            # OpenCV calls, and running them inline here would stall every
            # other client sharing this same uvicorn event loop for as
            # long as they take, exactly the kind of per-client isolation
            # the `get_latest()` offload above is already there for.
            jpeg_bytes = await asyncio.to_thread(self._render_jpeg, frame.image, snapshot)
            yield (
                b"--" + _BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(jpeg_bytes)).encode()
                + b"\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )

    def _render_jpeg(self, frame_image: np.ndarray, snapshot: DebugSnapshot) -> bytes:
        """Render + JPEG-encode one frame -- the synchronous, CPU-bound half
        of `_stream()`'s per-iteration work, split out so it can be run via
        `asyncio.to_thread` (see the call site's own comment).
        """
        annotated = render_overlay(
            frame_image,
            self._calibration,
            snapshot.tracked_frame,
            snapshot.frame_assignments,
            snapshot.state_snapshot,
        )
        return _encode_jpeg(annotated)

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


def build_debug_server(config: Config, calibration: CalibrationRuntime) -> MjpegDebugServer | None:
    """Construct the MJPEG debug server (and its `LatestFrameHub`) iff
    `config.debug.enabled` (REQ-37, REQ-46, AC-24).

    Returns `None` when disabled, so the caller never constructs a
    `FastAPI` app, a `LatestFrameHub`, or starts a uvicorn server for it --
    "debug.enabled: false startet keinen MJPEG-Endpoint" (AC-24), not
    merely one that draws nothing. The returned server's `.frame_hub` is
    what the loop (`runner/lifecycle.py`) publishes into (REQ-46) -- this
    function is the single place that hub gets created, so the loop and
    this server always share the exact same instance.
    """
    if not config.debug.enabled:
        return None
    return MjpegDebugServer(calibration, LatestFrameHub(), websocket_port=config.ports.websocket)
