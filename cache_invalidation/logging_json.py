"""结构化日志：每行一个 JSON 对象，含时间、业务码与出错键。"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": time.strftime(
                "%Y-%m-%dT%H:%M:%S",
                time.localtime(record.created)) + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for field in ("code", "key", "field_path", "line", "details"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logger(name: str = "cache_invalidation",
                     log_path: str | Path | None = None,
                     stream: Any | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = JsonLineFormatter()

    stream_handler = logging.StreamHandler(stream or sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def log_business_error(logger: logging.Logger,
                       error: Exception,
                       default_code: str = "E_UNKNOWN") -> None:
    payload = getattr(error, "to_dict", None)
    data = payload() if callable(payload) else {"code": default_code,
                                                "message": str(error)}
    logger.error(
        data.get("message", str(error)),
        extra={key: value for key, value in data.items()
               if key != "message"},
    )
