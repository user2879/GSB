"""Thread-safe structured JSONL logging."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import TunnelError


class StructuredLogger:
    def __init__(self, path: str | Path | None = None, *, echo: bool = False) -> None:
        self.path = Path(path) if path is not None else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._lock = threading.Lock()

    def event(
        self,
        event: str,
        *,
        code: str = "OK",
        message: str = "",
        **fields: Any,
    ) -> dict[str, Any]:
        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "monotonic_ms": time.monotonic_ns() // 1_000_000,
            "event": event,
            "code": code,
            "message": message,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self.path is not None:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
            if self.echo:
                print(line)
        return record

    def error(self, event: str, error: TunnelError | BaseException, **fields: Any) -> None:
        if isinstance(error, TunnelError):
            context = error.to_dict()
        else:
            context = {"code": "E_STATE", "message": str(error)}
        self.event(
            event,
            code=str(context.get("code", "E_STATE")),
            message=str(context.get("message", "")),
            error=context,
            **fields,
        )
