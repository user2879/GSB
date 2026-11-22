"""结构化 JSON 行日志。"""

import json
import os
import threading
import time


class JsonlLogger:
    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, event, code=None, **fields):
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "event": event,
            "code": code,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return record
