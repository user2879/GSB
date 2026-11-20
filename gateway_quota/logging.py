from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO


class StructuredLogger:
    """Deterministic JSONL logger with a process-local sequence number."""

    def __init__(self, path: str | Path | None = None, stream: TextIO | None = None):
        self.path = Path(path) if path is not None else None
        self.stream = stream
        self.sequence = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                with self.path.open("r", encoding="utf-8") as handle:
                    self.sequence = sum(1 for line in handle if line.strip())

    def emit(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        self.sequence += 1
        record = {
            "log_seq": self.sequence,
            "code": code,
            "message": message,
            **fields,
        }
        line = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
        if self.stream is not None:
            print(line, file=self.stream, flush=True)
        return record

    def validation_failure(self, error: Any) -> None:
        for record in error.records:
            payload = record.to_dict()
            self.emit(
                payload["code"],
                payload["message"],
                path=payload.get("path"),
                line=payload.get("line"),
                business_code=payload.get("business_code"),
                details=payload.get("details", {}),
            )


def default_logger(path: str | Path | None = None) -> StructuredLogger:
    return StructuredLogger(path=path, stream=sys.stderr)
