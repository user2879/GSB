"""结构化 JSONL 日志。

每行一条 JSON，字段固定，便于回放比对与离线分析；不依赖任何第三方库。
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional


class StructuredLogger:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            # 新建即清空，保证每次实验可复现
            open(path, "w", encoding="utf-8").close()

    def log(self, event: str, level: str = "INFO", **fields: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "ts_wall_ms": int(time.time() * 1000),
            "level": level,
            "event": event,
        }
        entry.update(fields)
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if self.path:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        return entry

    def read(self) -> list[dict[str, Any]]:
        if not self.path or not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
