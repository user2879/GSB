"""Structured, line-delimited JSON logging (one JSON object per line)."""

import json
import os
import threading
import time

_LOCK = threading.Lock()


class StructuredLogger:
    """Append-only JSON-lines logger used for injected-fault reporting."""

    def __init__(self, path=None):
        self.path = path
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)

    def log(self, level, event, **fields):
        record = {
            "ts_ms": int(time.time() * 1000.0),
            "level": level,
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        with _LOCK:
            if self.path:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            else:
                print(line)
        return record
