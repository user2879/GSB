"""Deterministic, append-only JSONL structured logger.

Every event gets a monotonic logical clock value and a per-event sequence
number.  Wall clock time is never written so identical inputs and injection
sequences produce byte-identical log output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, TextIO


class JsonlLogger:
    def __init__(self, log_path: str | Path, echo: bool = False) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = self.log_path.open("w", encoding="utf-8", newline="\n")
        self._seq = 0
        self._clock = 0
        self.echo = echo

    def event(self, name: str, **fields: Any) -> Dict[str, Any]:
        self._seq += 1
        self._clock += 1
        record: Dict[str, Any] = {
            "seq": self._seq,
            "clock": self._clock,
            "event": name,
        }
        for key in sorted(fields):
            record[key] = fields[key]
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self._handle.write(line + "\n")
        self._handle.flush()
        if self.echo:
            print(line)
        return record

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
