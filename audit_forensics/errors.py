"""Business error and configuration exception definitions."""

from __future__ import annotations

from typing import Any


E_MALFORMED_JSON = "E_MALFORMED_JSON"
E_MALFORMED_EVENT = "E_MALFORMED_EVENT"
E_SCHEMA = "E_SCHEMA"
E_TIMESTAMP = "E_TIMESTAMP"
E_ENCODING = "E_ENCODING"
E_EMPTY_INPUT = "E_EMPTY_INPUT"
E_FORMAT_UNKNOWN = "E_FORMAT_UNKNOWN"
E_IO = "E_IO"
E_ORPHAN = "E_ORPHAN"
E_CYCLE = "E_CYCLE"
E_CROSS_TRACE = "E_CROSS_TRACE"
E_EVENT_CONFLICT = "E_EVENT_CONFLICT"
E_SEQ_ROLLBACK = "E_SEQ_ROLLBACK"
E_SEQ_GAP = "E_SEQ_GAP"
E_SEQ_DUPLICATE = "E_SEQ_DUPLICATE"
E_LOCK = "E_LOCK"
E_CHECKPOINT_MISSING = "E_CHECKPOINT_MISSING"
E_CHECKPOINT_INVALID = "E_CHECKPOINT_INVALID"
E_CHECKPOINT_CORRUPT = "E_CHECKPOINT_CORRUPT"
E_STATE_SCHEMA_MISMATCH = "E_STATE_SCHEMA_MISMATCH"
E_RULES_INVALID = "E_RULES_INVALID"
E_RULES_VERSION_CHANGED = "E_RULES_VERSION_CHANGED"
E_RULES_CONFIG_CHANGED = "E_RULES_CONFIG_CHANGED"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_CASE_NOT_FOUND = "E_CASE_NOT_FOUND"
E_NO_STATE = "E_NO_STATE"
E_REPORT_WRITE = "E_REPORT_WRITE"
E_FORCED_REBUILD = "E_FORCED_REBUILD"


def business_error(
    code: str,
    message: str,
    *,
    source_format: str | None = None,
    source: str | None = None,
    position: str | int | None = None,
    event_id: str | None = None,
    trace_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-safe, stably keyed business error record."""

    result: dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if source_format is not None:
        result["source_format"] = source_format
    if source is not None:
        result["source"] = source
    if position is not None:
        result["position"] = position
    if event_id is not None:
        result["event_id"] = event_id
    if trace_id is not None:
        result["trace_id"] = trace_id
    if details is not None:
        result["details"] = details
    return result


class AuditConfigError(ValueError):
    """A command-line or rule configuration error (process exit code 2)."""

    def __init__(self, message: str, *, code: str = E_RULES_INVALID) -> None:
        super().__init__(message)
        self.code = code


class LockConflictError(RuntimeError):
    """The workspace write lock is already held."""

    code = E_LOCK
