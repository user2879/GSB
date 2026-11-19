"""Structured JSONL logger with stable key order and a console mirror."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, log_path: str | Path, echo: bool = True):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._lock = threading.Lock()

    def log(self, level: str, event: str, **fields: Any) -> dict[str, Any]:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "level": level,
            "event": event,
            **fields,
        }
        line = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            if self.echo:
                sys.stdout.write(line + "\n")
        return entry

    def info(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.log("INFO", event, **fields)

    def error(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.log("ERROR", event, **fields)
