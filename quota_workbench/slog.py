"""Structured JSON-lines logging."""

from __future__ import annotations

import json
import os
import threading
import time


class StructuredLogger:
    """Append-only JSONL logger; one object per event, stable key order."""

    def __init__(self, path: str | None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8", newline="\n")

    def log(self, event: str, code: str | None = None, **fields) -> None:
        record = {"ts_ms": int(time.monotonic() * 1000), "event": event}
        if code is not None:
            record["code"] = code
        for key in sorted(fields):
            record[key] = fields[key]
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "StructuredLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


NULL_LOGGER = StructuredLogger(None)
