"""结构化 JSONL 日志：每条事件一行，首次/重试回执分别成条。"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class StructuredLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")
        self._t0 = time.monotonic()

    def monotonic_ms(self) -> int:
        return int(round((time.monotonic() - self._t0) * 1000))

    def log(
        self,
        event: str,
        code: Optional[str] = None,
        **fields: Any,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "monotonic_ms": self.monotonic_ms(),
            "event": event,
        }
        if code:
            record["code"] = code
        record.update(fields)
        line = json.dumps(record, ensure_ascii=True, sort_keys=True)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
        return record

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()
