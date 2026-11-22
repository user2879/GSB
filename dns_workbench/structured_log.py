"""结构化日志：同时输出 JSON Lines 文件与可读控制台日志。"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .errors import WorkbenchError


class JsonLineHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if isinstance(extra, dict):
            payload.update(extra)
        with open(self.baseFilename, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def configure_logger(path: str | Path, *, name: str = "dns_workbench") -> logging.Logger:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    json_handler = JsonLineHandler(str(path), encoding="utf-8")
    text_handler = logging.StreamHandler()
    text_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(json_handler)
    logger.addHandler(text_handler)
    logger.propagate = False
    return logger


def log_error(logger: logging.Logger, exc: WorkbenchError, **context) -> None:
    data = exc.to_dict()
    data.update(context)
    logger.error("%s: %s", exc.code, exc.message, extra={"data": data})
