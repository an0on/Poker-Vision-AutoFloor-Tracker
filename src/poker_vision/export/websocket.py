"""`websocket` export adapter (REQ-35): FastAPI/uvicorn live event stream.

A connecting WebSocket client first receives one full `StateSnapshot`
(REQ-33) so it can catch up on everything that happened before it
connected, then every subsequent `Event` this adapter's `export()` is
given, in the exact order given -- the same order `JsonlEventExporter`
(REQ-34) writes to disk for the same session, so a client's view and the
JSONL file never disagree (AC-22).

`export()` is the same synchronous, fire-and-forget call the rest of the
pipeline already makes against `JsonlEventExporter`, but a WebSocket send
only happens inside the ASGI server's own event loop. Bridging the two
without binding this adapter to a specific event loop (uvicorn's in
production, the test client's portal thread in tests) is done with a plain
`queue.Queue` per connection: `export()` just enqueues JSON text from
whatever thread the pipeline runs on, and each connection's async handler
polls its own queue and forwards whatever it finds to the socket. No
`asyncio` object is ever touched from outside the loop that owns it.

A client is registered (queue created, added to `_connections`) before its
initial snapshot is sent, so an `export()` call that races the handshake
enqueues rather than gets lost -- the client may see one event twice
(once folded into the snapshot, once replayed) but never a gap, which
matches this adapter's job of never losing an event, not deduplicating a
narrow startup race.

Each connection's send loop also polls the ASGI receive channel (with the
same short timeout used for the queue poll) purely to notice a disconnect
promptly: this adapter never expects a client to send anything, but with
no receive() call at all, a client that disconnects while its queue is
empty would go unnoticed until some later `export()` happened to hit a
failed send -- or forever, if none ever does -- leaving its handler
coroutine and queue resident indefinitely.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import Iterable

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from poker_vision.state.events import Event
from poker_vision.state.machine import PipelineStateMachine

_QUEUE_POLL_INTERVAL_SECONDS = 0.01


class WebSocketEventExporter:
    """Serves live `Event`s plus REST status over FastAPI/uvicorn (REQ-35)."""

    def __init__(self, state_machine: PipelineStateMachine) -> None:
        self._state_machine = state_machine
        self._connections: list[queue.Queue[str]] = []
        self._lock = threading.Lock()
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/status")
        def status() -> dict:
            return json.loads(self._state_machine.snapshot().model_dump_json())

        @app.websocket("/ws")
        async def stream(websocket: WebSocket) -> None:
            await self._handle_connection(websocket)

    async def _handle_connection(self, websocket: WebSocket) -> None:
        await websocket.accept()
        client_queue: queue.Queue[str] = queue.Queue()
        with self._lock:
            self._connections.append(client_queue)
        try:
            # Sent only after registration, so any event exported from this
            # point on is queued rather than dropped while the snapshot
            # itself is still being serialized/sent.
            await websocket.send_text(self._state_machine.snapshot().model_dump_json())
            while True:
                try:
                    inbound = await asyncio.wait_for(
                        websocket.receive(), timeout=_QUEUE_POLL_INTERVAL_SECONDS
                    )
                except TimeoutError:
                    pass
                else:
                    if inbound["type"] == "websocket.disconnect":
                        break
                    # This is a one-way push stream -- any other inbound
                    # message (a client isn't expected to send one) is
                    # simply ignored rather than treated as an error.
                    continue

                try:
                    message = client_queue.get_nowait()
                except queue.Empty:
                    continue
                await websocket.send_text(message)
        except WebSocketDisconnect:
            pass
        finally:
            with self._lock:
                if client_queue in self._connections:
                    self._connections.remove(client_queue)

    def export(self, events: Iterable[Event]) -> None:
        messages = [event.model_dump_json() for event in events]
        if not messages:
            return
        with self._lock:
            connections = list(self._connections)
        for client_queue in connections:
            for message in messages:
                client_queue.put(message)

    def run(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """Block, serving this adapter's FastAPI app via uvicorn."""
        uvicorn.run(self.app, host=host, port=port)
