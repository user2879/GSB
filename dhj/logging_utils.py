"""Structured JSONL logging (one canonical-ish JSON object per line)."""

import json
import os
import threading
import time

_LOCK = threading.Lock()


class JsonlLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        open(self.path, "a", encoding="utf-8").close()

    def event(self, event, **fields):
        record = {
            "ts_ns": time.time_ns(),
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
        with _LOCK:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return record
