"""Structured JSON-lines event logging."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .errors import WorkbenchError


class JsonlEventLog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()

    def write(self, event: str, *, code: str | None = None, severity: str = "INFO", **fields: Any) -> dict[str, Any]:
        record = {
            "event": event,
            "severity": severity,
            "code": code,
            "time_unix_ms": int(time.time() * 1000),
        }
        for key, value in fields.items():
            if value is not None:
                record[key] = value
        line = canonical_json(record)
        if self.path is not None:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
        return record

    def error(self, event: str, error: WorkbenchError) -> dict[str, Any]:
        return self.write(
            event,
            code=error.code,
            severity="ERROR",
            message=error.message,
            revision=error.revision,
            field_path=error.field_path,
            layer=error.layer,
            details=error.details,
        )


def read_events(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events
