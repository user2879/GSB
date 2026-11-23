from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredLogger:
    """Append-only JSONL logger with a monotonic timestamp in every record."""

    def __init__(self, path: str | os.PathLike[str], component: str = "gateway-diag") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.component = component
        self._lock = threading.Lock()

    def emit(
        self,
        event: str,
        *,
        level: str = "INFO",
        code: str | None = None,
        rule_id: str | None = None,
        field: str | None = None,
        entry: int | str | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "mono_ns": time.monotonic_ns(),
            "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level,
            "component": self.component,
            "event": event,
        }
        if code is not None:
            record["code"] = code
        if rule_id is not None:
            record["rule_id"] = rule_id
        if field is not None:
            record["field"] = field
        if entry is not None:
            record["entry"] = entry
        if details:
            record["details"] = details
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return record

    def error(self, exc: Exception, event: str = "business_error") -> dict[str, Any] | None:
        from .errors import GatewayError

        if not isinstance(exc, GatewayError):
            return None
        return self.emit(
            event,
            level="ERROR",
            code=exc.code,
            rule_id=exc.rule_id,
            field=exc.field,
            entry=exc.entry,
            message=exc.message,
            **exc.details,
        )


def default_log_path(workdir: str | os.PathLike[str]) -> Path:
    return Path(workdir) / "events.jsonl"
