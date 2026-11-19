"""Structured JSONL logging with process-safe append semantics."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .util import mono_ms, wall_time_utc


class JsonlLogger:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, event: str, **fields: Any) -> None:
        record = {
            "event": event,
            "mono_ms": mono_ms(),
            "timestamp_utc": wall_time_utc(),
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if self.path is None:
            return
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line + "\n")
                stream.flush()

