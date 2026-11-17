"""Structured JSON-lines logger used for business-code audit trails."""

import json
import os
import time


class StructuredLogger:
    def __init__(self, path):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)

    def log(self, code, message, *, subscriber_id=None, event_seq=None, **extra):
        record = {
            "ts_ms": time.time_ns() // 1_000_000,
            "code": code,
            "subscriber_id": subscriber_id,
            "event_seq": event_seq,
            "message": message,
        }
        for key, value in extra.items():
            record[key] = value
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return record
