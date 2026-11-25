from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import monotonic_ms
from .errors import LedgerError


class StructuredLogger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, code: str, message: str, level: str = "INFO", **details: Any) -> dict[str, Any]:
        record = {
            "monotonic_ms": monotonic_ms(),
            "wall_utc": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "code": code,
            "message": message,
            **details,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        print(line)
        return record

    def log_error(self, error: LedgerError | Exception, **details: Any) -> dict[str, Any]:
        if isinstance(error, LedgerError):
            return self.log(error.code, str(error), level="ERROR", error=error.to_dict(), **details)
        return self.log("E_UNEXPECTED", str(error), level="ERROR", error_type=type(error).__name__, **details)
