"""Structured JSON-lines logger for audits and fault-injection evidence."""

import json
import os
import threading
import time


class StructuredLogger:
    def __init__(self, log_dir: str, name: str = "workbench"):
        os.makedirs(log_dir, exist_ok=True)
        day = time.strftime("%Y%m%d")
        self.path = os.path.join(log_dir, f"{name}-{day}.jsonl")
        self._lock = threading.Lock()

    def log(self, event: str, **fields):
        record = {"ts_ms": int(time.time() * 1000), "event": event, **fields}
        line = json.dumps(record, sort_keys=True, ensure_ascii=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return record
