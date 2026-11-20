from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_json_text
from .errors import InteropError, fail


class StructuredLogger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None

    def emit(
        self,
        *,
        business_code: str,
        message: str,
        manifest_path: str | None = None,
        shard_index: int | None = None,
        operation: str | None = None,
        level: str = "ERROR",
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level,
            "business_code": business_code,
            "message": message,
            "manifest_path": manifest_path,
            "shard_index": shard_index,
            "operation": operation,
            "details": details or {},
        }
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(canonical_json_text(event))
                    handle.write("\n")
            except OSError as exc:
                fail("E_IO", f"cannot write structured log: {exc}", manifest_path=str(self.path))
        return event

    def info(
        self,
        message: str,
        *,
        operation: str | None = None,
        manifest_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.emit(
            business_code="OK",
            message=message,
            manifest_path=manifest_path,
            operation=operation,
            level="INFO",
            details=details,
        )

    def error(self, error: InteropError, *, operation: str | None = None) -> dict[str, Any]:
        return self.emit(
            business_code=error.code,
            message=error.message,
            manifest_path=error.manifest_path,
            shard_index=error.shard_index,
            operation=operation,
            level="ERROR",
            details={"path": error.path, **error.details},
        )
