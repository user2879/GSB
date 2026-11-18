from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json, utc_now
from .errors import PipelineError


class StructuredLogger:
    def __init__(self, path: str | Path | None) -> None:
        self.path = None if path is None else Path(path)
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        event: str,
        *,
        code: str | None = None,
        message: str | None = None,
        location: str | None = None,
        task_id: str | None = None,
        field_name: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if self.path is None:
            return
        payload: dict[str, Any] = {
            "timestamp": utc_now(),
            "event": event,
        }
        if code is not None:
            payload["code"] = code
        if message is not None:
            payload["message"] = message
        if location is not None:
            payload["location"] = location
        if task_id is not None:
            payload["task_id"] = task_id
        if field_name is not None:
            payload["field"] = field_name
        if details:
            payload["details"] = dict(details)
        line = canonical_json(payload) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()

    def error(self, error: PipelineError, *, event: str = "business_error") -> None:
        self.emit(
            event,
            code=error.code,
            message=error.message,
            location=error.location,
            task_id=error.task_id,
            field_name=error.field_name,
            details=error.details,
        )
