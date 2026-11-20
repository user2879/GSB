"""结构化 JSONL 日志。

每行是一条独立 JSON，字段顺序固定，时间采用单调时钟毫秒与 UTC 墙钟时间，
保证写入文件后可被确定性工具逐行解析。异常注入与调度事件都会落盘。
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional


class StructuredLogger:
    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._fh = None
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
            + f".{int((time.time() % 1) * 1000):03d}Z",
            "mono_ms": time.monotonic_ns() // 1_000_000,
            "event": event,
        }
        for key in sorted(fields):
            record[key] = fields[key]
        line = json.dumps(record, ensure_ascii=False, sort_keys=False)
        if self._fh is None:
            return
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "StructuredLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
