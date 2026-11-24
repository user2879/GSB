from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from .errors import AppError
from .hashing import canonical_text


class JsonLogger:
    def __init__(self, log_path: Path | None = None, *, stream: bool = True):
        self.log_path = Path(log_path) if log_path else None
        self.stream = stream
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, level: str, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "wall_ms": int(time.time() * 1000),
            "level": level,
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self.stream:
            print(line, file=sys.stderr, flush=True)
        return record

    def error(self, error: AppError | Exception, **fields: Any) -> dict[str, Any]:
        if isinstance(error, AppError):
            payload = error.to_dict()
        else:
            payload = {"code": "E_INTERNAL", "message": str(error), "locator": {}}
        merged = {"error": payload, **fields}
        return self.event("ERROR", "business_error", **merged)


def print_json(value: Any) -> None:
    print(canonical_text(value), flush=True)
