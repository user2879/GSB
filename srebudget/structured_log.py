"""Small append-only NDJSON logger for diagnostic events."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import SreError


class StructuredLogger:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        event: str,
        *,
        code: str | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        level: str = "info",
    ) -> dict[str, Any]:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level,
            "event": event,
            "code": code,
            "message": message,
            "details": details or {},
        }
        if self.path is not None:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            with self._lock:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
        return record

    def error(self, event: str, error: SreError) -> dict[str, Any]:
        return self.write(
            event,
            code=error.code,
            message=error.message,
            details=error.details,
            level="error",
        )
