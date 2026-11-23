"""Structured JSONL logging (one JSON object per line)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path


class JsonLineHandler(logging.Handler):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8", newline="\n")

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts_ms": time.time_ns() // 1_000_000,
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            payload["data"] = record.extra_data
        self._fh.write(json.dumps(payload, sort_keys=True,
                                  ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
        super().close()


def configure_logging(log_dir: str | Path) -> logging.Logger:
    path = Path(log_dir) / "pipeline.jsonl"
    logger = logging.getLogger("alert_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = JsonLineHandler(path)
    logger.addHandler(handler)
    return logger
