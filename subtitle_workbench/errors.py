"""Business error codes shared by the engine, controller and UI."""
from __future__ import annotations

from dataclasses import dataclass


class WorkbenchError(Exception):
    """Base error carrying a stable machine-readable code."""

    code = "E_UNKNOWN"

    def __init__(self, message: str = "", *, conflict_item_id: str | None = None):
        super().__init__(message or self.code)
        self.conflict_item_id = conflict_item_id


class DragBoundsError(WorkbenchError):
    """Target lane/frame is outside the declared safe area."""

    code = "E_DRAG_BOUNDS"


class TimeInvalidError(WorkbenchError):
    """start_frame >= end_frame or cue outside the timeline."""

    code = "E_TIME_INVALID"


class TrackConflictError(WorkbenchError):
    """Unresolvable overlap with a higher-priority, non-yielding item."""

    code = "E_TRACK_CONFLICT"


class WrapError(WorkbenchError):
    """Text cannot satisfy the declared line/width constraints."""

    code = "E_WRAP_IMPOSSIBLE"


class StaleVersionError(WorkbenchError):
    """A write references an outdated snapshot version."""

    code = "E_STALE_VERSION"


@dataclass(frozen=True)
class ErrorRecord:
    """UI-facing rejection payload; state never changes when one is produced."""

    code: str
    message: str
    conflict_item_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "conflict_item_id": self.conflict_item_id,
        }
