"""结构化日志：单行 JSON，业务码与关键依据统一字段。"""
from __future__ import annotations

import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S",
                                time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in ("name", "msg", "args", "levelname", "levelno",
                           "pathname", "filename", "module", "exc_info",
                           "exc_text", "stack_info", "lineno", "funcName",
                           "created", "msecs", "relativeCreated", "thread",
                           "threadName", "processName", "process", "taskName"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(level: str = "INFO", logfile: str | None = None) -> logging.Logger:
    logger = logging.getLogger("degprop")
    logger.handlers.clear()
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    if logfile:
        file_handler = logging.FileHandler(logfile, encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger
