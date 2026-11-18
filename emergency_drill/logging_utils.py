"""Deterministic JSONL logging with explicit logical-clock provenance."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .errors import BusinessError


class StructuredLogger:
    def __init__(self, path: str | Path, *, echo: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._lock = threading.Lock()
        self._sequence = 0

    def emit(
        self,
        event: str,
        *,
        business_code: str | None = None,
        stage_seq: int | None = None,
        field_path: str | None = None,
        logical_clock_ms: int = 0,
        clock_source: str = "logical",
        **fields: Any,
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            record: dict[str, Any] = {
                "log_seq": self._sequence,
                "event": event,
                "business_code": business_code,
                "stage_seq": stage_seq,
                "field_path": field_path,
                "logical_clock_ms": int(logical_clock_ms),
                "clock_source": clock_source,
            }
            record.update(fields)
            line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
            if self.echo:
                print(line)
            return record

    def error(
        self,
        error: BusinessError,
        *,
        event: str = "business_error",
        logical_clock_ms: int = 0,
        **fields: Any,
    ) -> dict[str, Any]:
        extra = {
            "message": error.message,
            "error_details": error.details,
            **fields,
        }
        return self.emit(
            event,
            business_code=error.code,
            stage_seq=error.stage_seq,
            field_path=error.field_path,
            logical_clock_ms=logical_clock_ms,
            **extra,
        )
