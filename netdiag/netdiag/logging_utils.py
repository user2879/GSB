"""结构化 JSON 行日志。"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, TextIO


class StructuredLogger:
    """每行一条 JSON，字段固定含 ts/level/event，便于机器消费。"""

    def __init__(self, path: Optional[str | Path] = None,
                 stream: Optional[TextIO] = None):
        self._lock = threading.Lock()
        self._file = open(path, "a", encoding="utf-8") if path else None
        self._stream = stream or sys.stderr

    def log(self, level: str, event: str, **fields: Any) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S",
                                time.localtime()) + f".{time.time_ns() // 1000 % 1000:03d}",
            "level": level,
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
        with self._lock:
            if self._file:
                self._file.write(line + "\n")
                self._file.flush()
            if self._stream:
                self._stream.write(line + "\n")
        return record

    def info(self, event: str, **fields: Any) -> Dict[str, Any]:
        return self.log("INFO", event, **fields)

    def warn(self, event: str, **fields: Any) -> Dict[str, Any]:
        return self.log("WARN", event, **fields)

    def error(self, event: str, **fields: Any) -> Dict[str, Any]:
        return self.log("ERROR", event, **fields)

    def close(self) -> None:
        if self._file:
            self._file.close()
