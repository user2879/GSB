"""Minimal structured JSONL logging with no external services."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes
from .errors import SlowProfError


class StructuredLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _write(self, record: dict[str, Any]) -> None:
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **record}
        line = canonical_json_bytes(record) + b"\n"
        with self._lock:
            with open(self.path, "ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    def event(self, event: str, **fields: Any) -> None:
        self._write({"event": event, **fields})

    def error(self, error: SlowProfError, **context: Any) -> None:
        record: dict[str, Any] = {
            "event": "error",
            "ok": False,
            "code": error.code,
            "message": error.message,
        }
        if error.family is not None:
            record["family"] = error.family
        if error.line_no is not None:
            record["line"] = error.line_no
        if error.field is not None:
            record["field"] = error.field
        if error.details:
            record["details"] = error.details
        record.update(context)
        self._write(record)


class NullLogger:
    def event(self, event: str, **fields: Any) -> None:
        return None

    def error(self, error: SlowProfError, **context: Any) -> None:
        return None

