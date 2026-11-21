"""Structured JSON-lines logging for the workbench."""

from __future__ import annotations

import json
import os
import threading
import time


class StructuredLogger:
    def __init__(self, log_dir="logs", name="run"):
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(log_dir, f"lbhc-{name}-{stamp}.jsonl")
        self._lock = threading.Lock()

    def emit(self, event, **fields):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "event": event, **fields}
        line = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return rec
