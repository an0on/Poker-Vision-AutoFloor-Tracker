"""Common `Capture` interface implemented by `continuity`, `video_file` and
`image_dir` (REQ-13).

All three yield the same `Frame` shape through the same iterator protocol,
so downstream pipeline stages never know which concrete source produced a
frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from poker_vision.capture.frame import Frame


class Capture(ABC):
    """Iterable source of `Frame`s. Exhausting the iterator or calling
    `close()` releases any underlying resource (file handle, camera device).
    """

    source_id: str

    def __iter__(self) -> Capture:
        return self

    @abstractmethod
    def __next__(self) -> Frame:
        """Return the next frame, or raise `StopIteration` when the source
        is exhausted (`video_file`/`image_dir` only — `continuity` never
        raises `StopIteration`; a missing/failed camera is a hard error,
        not exhaustion, see REQ-16)."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying resource. Safe to call more than once."""

    def __enter__(self) -> Capture:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
