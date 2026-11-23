"""Append-only JSON-lines structured logging with monotonic timestamps."""

from __future__ import annotations

import datetime as _datetime
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class StructuredLogger:
    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        event: str,
        *,
        code: str | None = None,
        edit_seq: int | None = None,
        node_id: str | None = None,
        details: dict[str, Any] | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        monotonic_ns = time.monotonic_ns()
        record: dict[str, Any] = {
            "mono_ns": monotonic_ns,
            "utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "event": event,
            "code": code,
            "edit_seq": edit_seq,
            "node_id": node_id,
            "details": details or {},
        }
        record.update(fields)
        if self.path is not None:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            with self._lock:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
        return record

    def error(self, error: Any, event: str, *, edit_seq: int | None = None, **fields: Any) -> dict[str, Any]:
        details = error.to_dict() if hasattr(error, "to_dict") else {"message": str(error)}
        return self.emit(
            event,
            code=getattr(error, "code", "E_INTERNAL"),
            edit_seq=getattr(error, "edit_seq", None) if edit_seq is None else edit_seq,
            node_id=getattr(error, "node_id", None),
            details=details,
            **fields,
        )
