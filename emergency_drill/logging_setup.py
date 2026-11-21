"""结构化 JSON Lines 日志（标准库 logging，无第三方依赖）。"""

import json
import logging
import os


class JsonLineFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = getattr(record, "context", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(log_path, level=logging.INFO):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    logger = logging.getLogger("drill")
    logger.setLevel(level)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger():
    return logging.getLogger("drill")


def log_business_error(logger, error, action):
    context = error.as_dict() if hasattr(error, "as_dict") else {"message": str(error)}
    logger.error("business rule rejected", extra={"context": {"action": action, **context}})
