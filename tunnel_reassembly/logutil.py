"""结构化 JSONL 日志，每行一个 JSON 对象，渲染/读取顺序稳定。"""

from __future__ import annotations

import json
import time
from pathlib import Path


class StructuredLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a", encoding="utf-8")
        self._seq = sum(1 for _ in open(self.path, "r", encoding="utf-8"))

    def event(self, level: str, code: str, message: str, **fields) -> dict:
        self._seq += 1
        record = {
            "seq": self._seq,
            "ts_ms": time.monotonic_ns() // 1_000_000,
            "level": level,
            "code": code,
            "message": message,
        }
        record.update(sorted(fields.items()))
        self._handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        self._handle.flush()
        return record

    def close(self) -> None:
        self._handle.close()
