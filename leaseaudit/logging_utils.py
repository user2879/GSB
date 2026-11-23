from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TextIO

from .errors import BusinessError


class StructuredLogger:
    def __init__(
        self,
        path: str | Path | None = None,
        stream: TextIO | None = None,
        *,
        echo: bool = True,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.stream = stream if stream is not None else (sys.stderr if echo else None)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def emit(self, level: str, code: str, message: str, context: Mapping[str, Any] | None = None) -> dict:
        record = {
            "ts_utc": self._utc_now(),
            "level": level,
            "code": code,
            "message": message,
            "context": dict(context or {}),
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
        if self.stream is not None:
            print(line, file=self.stream)
        return record

    def info(self, code: str, message: str, **context: Any) -> dict:
        return self.emit("INFO", code, message, context)

    def error(self, error: BusinessError) -> dict:
        return self.emit("ERROR", error.code.value, error.message, error.to_dict())
