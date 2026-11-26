from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .util import monotonic_ms, wall_time_iso


class StructuredLogger:
    """Append-only JSONL logger with a monotonic clock in milliseconds."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, code: str, message: str, **context: Any) -> dict[str, Any]:
        event: dict[str, Any] = {
            "monotonic_ms": monotonic_ms(),
            "wall_time": wall_time_iso(),
            "code": code,
            "message": message,
        }
        if context:
            event["context"] = context
        with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return event
