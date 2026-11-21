from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, log_dir: str | Path, name: str = "streamagg") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{name}.jsonl"
        self._lock = threading.Lock()
        self._sequence = 0

    def emit(
        self,
        *,
        level: str,
        code: str,
        event: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            payload = {
                "sequence": self._sequence,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "level": level,
                "code": code,
                "event": event,
                "message": message,
                "details": details or {},
            }
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
                handle.flush()
            return payload

    def error(self, error: Any, event: str = "business_error") -> dict[str, Any]:
        if hasattr(error, "as_dict"):
            payload = error.as_dict()
            return self.emit(
                level="ERROR",
                code=payload["code"],
                event=event,
                message=payload["message"],
                details={key: value for key, value in payload.items() if key not in {"code", "message"}},
            )
        return self.emit(
            level="ERROR",
            code="E_INTERNAL",
            event=event,
            message=str(error),
            details={"type": type(error).__name__},
        )

    def info(self, code: str, event: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.emit(level="INFO", code=code, event=event, message=message, details=details)
