"""Structured JSON-lines logging. One event per line, stable key order."""

import json
import os
import threading
import time


class StructLog:
    def __init__(self, path):
        self.path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event, **fields):
        record = {"ts": time.time(), "event": event}
        record.update(fields)
        line = json.dumps(record, sort_keys=True, ensure_ascii=True)
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def error(self, code, message, **fields):
        self.emit("error", code=code, message=message, **fields)
