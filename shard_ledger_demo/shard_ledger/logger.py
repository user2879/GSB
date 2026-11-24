"""Structured JSONL logger."""

import json
import os

from .clock import now_ms, wall_iso


class StructuredLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._handle = open(path, "w", encoding="utf-8")
        self._counter = 0

    def event(self, name: str, **fields):
        self._counter += 1
        record = {
            "log_seq": self._counter,
            "wall_ts": wall_iso(),
            "mono_ms": now_ms(),
            "event": name,
            **fields,
        }
        self._handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        self._handle.flush()

    def error(self, error):
        self._counter += 1
        record = {
            "log_seq": self._counter,
            "wall_ts": wall_iso(),
            "mono_ms": now_ms(),
            "event": "error",
            **error.to_dict(),
        }
        self._handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self):
        if not self._handle.closed:
            self._handle.close()
