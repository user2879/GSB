"""Structured JSON-lines logging for CLI and injected failures."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .util import wall_time_ms


class StructuredLogger:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event: str, **fields: Any) -> None:
        if self.path is None:
            return
        record = {
            "timestamp_ms": wall_time_ms(),
            "event": event,
            **fields,
        }
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")

