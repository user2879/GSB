"""结构化业务错误码。"""

from __future__ import annotations


class DiagError(Exception):
    """携带稳定业务码的诊断错误。"""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}
