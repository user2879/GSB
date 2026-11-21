"""结构化 JSONL 日志（每行一个 JSON 对象，键顺序稳定）。"""

import json
import os
import time


class StructuredLogger:
    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, "events.jsonl")
        os.makedirs(out_dir, exist_ok=True)
        self._handle = open(self.path, "a", encoding="utf-8")

    def log(self, level, event, **fields):
        record = {
            "ts_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "level": level,
            "event": event,
        }
        record.update(fields)
        self._handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self):
        self._handle.close()
