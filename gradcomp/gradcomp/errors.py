"""业务错误码与统一异常。

所有可预期的业务失败都抛出 :class:`GradCompError`，携带稳定的错误码，
CLI 入口负责打印 JSON 形式的错误对象并返回对应的退出码。
"""
from __future__ import annotations

from typing import Any


E_CONFIG = "E_CONFIG"
E_PORT = "E_PORT"
E_PROTO = "E_PROTO"
E_DIVERGE = "E_DIVERGE"
E_REPLAY = "E_REPLAY"
E_SNAPSHOT = "E_SNAPSHOT"
E_INTERNAL = "E_INTERNAL"

EXIT_CODES = {
    E_CONFIG: 2,
    E_PORT: 3,
    E_PROTO: 4,
    E_DIVERGE: 5,
    E_REPLAY: 6,
    E_SNAPSHOT: 7,
    E_INTERNAL: 70,
}


class GradCompError(Exception):
    """带稳定业务码的异常。"""

    def __init__(self, code: str, message: str, field: str | None = None,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": self.code, "message": self.message}
        if self.field is not None:
            payload["field"] = self.field
        if self.details:
            payload["details"] = self.details
        return payload
