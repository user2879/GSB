from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


class EventLogger:
    """Append-only JSON Lines logger with a lock for worker-thread safety."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(
        self,
        event: str,
        *,
        code: str | None = None,
        object_key: str | None = None,
        shard_index: int | None = None,
        message: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "monotonic_ms": monotonic_ms(),
            "event": event,
        }
        if code is not None:
            record["code"] = code
        if object_key is not None:
            record["object_key"] = object_key
        if shard_index is not None:
            record["shard_index"] = shard_index
        if message is not None:
            record["message"] = message
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
        return record

    def business_error(self, error: Any, **fields: Any) -> dict[str, Any]:
        detail = getattr(error, "detail", {}) or {}
        merged = dict(detail)
        merged.update(fields)
        return self.emit(
            "business_error",
            code=getattr(error, "code", "E_UNKNOWN"),
            object_key=getattr(error, "object_key", None),
            shard_index=getattr(error, "shard_index", None),
            message=getattr(error, "message", str(error)),
            field=getattr(error, "field", None),
            **merged,
        )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
