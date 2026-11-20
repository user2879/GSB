"""结构化日志: JSONL, 含记录标识、字段与检出阶段."""

import json
import os
import time


class StructuredLogger:
    def __init__(self, path):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.events = []

    def event(self, code, stage, record_id=None, field=None, message="", **extra):
        entry = {
            "ts": time.time(),
            "code": code,
            "stage": stage,
            "record_id": record_id,
            "field": field,
            "message": message,
        }
        if extra:
            entry["extra"] = extra
        self.events.append(entry)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
        return entry
