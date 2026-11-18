"""结构化 JSONL 日志：每行一个 JSON 对象，便于诊断与异常注入留痕。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, level: str, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False,
                                    sort_keys=True) + "\n")

    def info(self, event: str, **fields: Any) -> None:
        self.log("INFO", event, **fields)

    def error(self, event: str, code: str, **fields: Any) -> None:
        self.log("ERROR", event, code=code, **fields)
