"""Structured business exceptions and JSONL diagnostics."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import monotonic_ms


ERROR_CODES = {
    "E_SPEC": "invalid input specification",
    "E_GRAPH": "invalid shard dependency graph",
    "E_PORT": "node port unavailable",
    "E_TIMEOUT": "real socket timeout",
    "E_REPLY": "invalid node reply",
    "E_LEASE": "lease validation failed",
    "E_REPLAY": "replay comparison failed",
    "E_SNAPSHOT": "ledger snapshot validation failed",
    "E_LEDGER": "ledger operation failed",
}


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    position: str | None = None
    field: str | None = None
    shard_id: str | None = None
    event_seq: int | None = None
    detail: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self, *, mono_ms: int | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "monotonic_ms": monotonic_ms() if mono_ms is None else mono_ms,
            "position": self.position,
            "field": self.field,
            "shard_id": self.shard_id,
            "event_seq": self.event_seq,
        }
        if self.detail:
            item["detail"] = self.detail
        return item


class StructuredLogger:
    """Append-only JSONL log; each record is flushed before returning."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(
        self,
        code: str,
        message: str,
        *,
        shard_id: str | None = None,
        event_seq: int | None = None,
        position: str | None = None,
        field: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "monotonic_ms": monotonic_ms(),
            "code": code,
            "message": message,
            "shard_id": shard_id,
            "event_seq": event_seq,
            "position": position,
            "field": field,
        }
        if extra:
            record.update(extra)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
        return record

    def write_error(self, error: BusinessError) -> dict[str, Any]:
        return self.write(
            error.code,
            error.message,
            shard_id=error.shard_id,
            event_seq=error.event_seq,
            position=error.position,
            field=error.field,
            extra={"detail": error.detail} if error.detail else None,
        )
