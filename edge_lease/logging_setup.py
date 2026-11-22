"""Structured JSON-lines logging.

Every log record is a single JSON object on its own line written to a file
(and optionally mirrored to stderr).  Records are ordered by a monotonic
millisecond clock so timing analysis never depends on wall-clock adjustments.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


def mono_ms() -> int:
    """Monotonic clock truncated to integer milliseconds."""
    return int(time.monotonic() * 1000)


class JsonlLogger:
    """Thread-safe append-only JSON-lines structured logger."""

    def __init__(self, path: Optional[Path] = None, echo: bool = False) -> None:
        self.path = Path(path) if path is not None else None
        self.echo = echo
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.touch(exist_ok=True)

    def event(self, level: str, event: str, **fields: Any) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "ts_mono_ms": mono_ms(),
            "level": level,
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            if self.echo:
                sys.stderr.write(line + "\n")
                sys.stderr.flush()
        return record

    def info(self, event: str, **fields: Any) -> Dict[str, Any]:
        return self.event("INFO", event, **fields)

    def warning(self, event: str, **fields: Any) -> Dict[str, Any]:
        return self.event("WARN", event, **fields)

    def error(self, event: str, **fields: Any) -> Dict[str, Any]:
        return self.event("ERROR", event, **fields)

    def read_all(self):
        if self.path is None:
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
