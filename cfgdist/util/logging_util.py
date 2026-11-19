"""Structured, order-stable JSON-lines logger.

Each log line is a JSON object with monotonically increasing ``seq`` numbers.
Every business record carries either ``key`` (configuration item key / node id /
request id) or ``source`` (the triggering component or stage).
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .clock import now_ms


@dataclass
class StructuredLogger:
    path: Path | None = None
    _fh: TextIO | None = None
    _lock: threading.Lock = None  # type: ignore[assignment]
    _seq: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    def log(
        self,
        event: str,
        *,
        code: str | None = None,
        key: str | None = None,
        source: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        if key is None and source is None:
            source = source or "platform"
        record: dict[str, Any] = {
            "seq": None,
            "ts_ms": now_ms(),
            "event": event,
            "code": code,
            "key": key,
            "source": source,
        }
        record.update(fields)
        with self._lock:
            self._seq += 1
            record["seq"] = self._seq
            line = json.dumps(record, sort_keys=True, ensure_ascii=False)
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()
        return record

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "StructuredLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
