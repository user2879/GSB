"""结构化日志：JSON Lines 输出，可写入文件并可选回显到 stderr。"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonLogger:
    def __init__(self, path: str | Path | None = None, echo: bool = False):
        self.path = Path(path) if path else None
        self.echo = echo
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, level: str, code: str, message: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "code": code,
            "message": message,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        if self.echo:
            print(line, file=sys.stderr)
        return record

    def error(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        return self.emit("error", code, message, **fields)

    def info(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        return self.emit("info", code, message, **fields)
