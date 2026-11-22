"""结构化 JSONL 日志。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonlLogger:
    """每行一个 JSON 对象的结构化日志器，线程安全。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def log(self, level: str, event: str, code: str = "", **fields: Any) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            + f".{time.time_ns() % 1_000_000_000:09d}",
            "level": level,
            "event": event,
            "code": code,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        self._handle.write(line + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
