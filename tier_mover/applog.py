"""结构化 JSONL 日志。

每行一个 JSON 对象，包含时间戳、级别、事件、业务码与附加上下文。
异常注入事件与错误报告统一走该日志，保证可审计。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, log_path: str | Path | None = None):
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []

    def emit(self, level: str, event: str, **fields: Any) -> dict[str, Any]:
        entry = {
            "ts_ms": int(time.time() * 1000),
            "level": level,
            "event": event,
            **fields,
        }
        with self._lock:
            self._buffer.append(entry)
            if self.log_path:
                with self.log_path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
                    fh.write("\n")
        return entry

    def info(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.emit("INFO", event, **fields)

    def warn(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.emit("WARN", event, **fields)

    def error(self, event: str, **fields: Any) -> dict[str, Any]:
        return self.emit("ERROR", event, **fields)

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._buffer)
