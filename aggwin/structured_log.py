"""Structured JSONL event log."""

import json
import os
import time


class StructuredLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def log(self, level, event, message="", **context):
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "level": level, "event": event, "message": message,
                  "context": context}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return record

    def anomaly(self, code, message, **context):
        return self.log("ERROR", "anomaly", message, code=code, **context)
