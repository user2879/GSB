"""Stable error objects used by the command line, GUI, and tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolError(Exception):
    """A machine-readable local verification error.

    The stable fields deliberately mirror the attribution requirements:
    declaration path, frame type/offset, stream id, and replay frame index.
    """

    code: str
    message: str
    path: str | None = None
    frame_type: str | None = None
    offset: int | None = None
    frame_index: int | None = None
    stream_id: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            result["path"] = self.path
        if self.frame_type is not None:
            result["frame_type"] = self.frame_type
        if self.offset is not None:
            result["offset"] = self.offset
        if self.frame_index is not None:
            result["frame_index"] = self.frame_index
        if self.stream_id is not None:
            result["stream_id"] = self.stream_id
        if self.details:
            result["details"] = self.details
        return result

    def __str__(self) -> str:
        parts = [self.code]
        if self.path:
            parts.append(f"path={self.path}")
        if self.frame_type:
            parts.append(f"frame={self.frame_type}")
        if self.offset is not None:
            parts.append(f"offset={self.offset}")
        if self.frame_index is not None:
            parts.append(f"frame_index={self.frame_index}")
        if self.stream_id is not None:
            parts.append(f"stream={self.stream_id}")
        return f"{': '.join(parts)}: {self.message}"


class DeclarationError(ToolError):
    def __init__(self, message: str, path: str, **kwargs: Any) -> None:
        super().__init__("E_DECL", message, path=path, **kwargs)


class FrameError(ToolError):
    def __init__(
        self,
        message: str,
        *,
        frame_type: str = "UNKNOWN",
        offset: int = 0,
        path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            "E_FRAME",
            message,
            path=path,
            frame_type=frame_type,
            offset=offset,
            **kwargs,
        )


class StreamError(ToolError):
    def __init__(self, message: str, stream_id: int, **kwargs: Any) -> None:
        super().__init__("E_STREAM", message, stream_id=stream_id, **kwargs)


class ReplayError(ToolError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__("E_REPLAY", message, **kwargs)


class SnapshotError(ToolError):
    def __init__(self, message: str, path: str | None = None, **kwargs: Any) -> None:
        super().__init__("E_SNAPSHOT", message, path=path, **kwargs)
