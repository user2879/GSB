"""Small JSON-lines structured logger with deterministic field ordering."""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import AppError


class JsonLogger:
    def __init__(self, log_dir: str | os.PathLike[str] | None = None, *, echo: bool = False) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.echo = echo
        self._lock = threading.Lock()
        self.path: Path | None = None
        if self.log_dir is not None:
            self._prepare()

    def _prepare(self) -> None:
        assert self.log_dir is not None
        with self._lock:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                probe = self.log_dir / f".write-probe-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
            except OSError as exc:
                raise AppError("E_DATA", f"日志目录不可写: {self.log_dir}: {exc}", {"log_dir": str(self.log_dir)}) from exc
            self.path = self.log_dir / "vecflow.jsonl"

    def log(self, event: str, *, level: str = "INFO", code: str | None = None, message: str = "", **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level,
            "event": event,
        }
        if code:
            record["code"] = code
        if message:
            record["message"] = message
        for key in sorted(fields):
            value = fields[key]
            if isinstance(value, Path):
                value = str(value)
            record[key] = value
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if self.path is not None:
            with self._lock:
                try:
                    with self.path.open("a", encoding="utf-8") as handle:
                        handle.write(line + "\n")
                except OSError as exc:
                    raise AppError("E_DATA", f"写入日志失败: {self.path}: {exc}") from exc
        if self.echo:
            print(line, file=sys.stderr)
        return record

    def error(self, event: str, error: AppError, **fields: Any) -> dict[str, Any]:
        return self.log(event, level="ERROR", code=error.code, message=error.message, details=error.details, **fields)
