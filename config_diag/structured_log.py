from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlLogger:
    """Small append-only structured logger for local diagnostics."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(
        self,
        code: str,
        event: str,
        severity: str = "INFO",
        message: str = "",
        key: str | None = None,
        field: str | None = None,
        **context: Any,
    ) -> dict[str, Any]:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            "code": code,
            "event": event,
            "severity": severity,
            "message": message,
            "context": context,
        }
        if key is not None:
            record["key"] = key
        if field is not None:
            record["field"] = field
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return record


class NullLogger:
    def emit(
        self,
        code: str,
        event: str,
        severity: str = "INFO",
        message: str = "",
        key: str | None = None,
        field: str | None = None,
        **context: Any,
    ) -> dict[str, Any]:
        record = {
            "code": code,
            "event": event,
            "severity": severity,
            "message": message,
            "context": context,
        }
        if key is not None:
            record["key"] = key
        if field is not None:
            record["field"] = field
        return record
