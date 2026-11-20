"""Business error codes and structured exception type."""

from __future__ import annotations


class ErrorCode:
    E_FORMAT = "E_FORMAT"
    E_PORT = "E_PORT"
    E_VERSION = "E_VERSION"
    E_CONVERGE = "E_CONVERGE"
    E_IDEMPOTENT = "E_IDEMPOTENT"
    E_REPLAY = "E_REPLAY"
    E_STATE = "E_STATE"
    E_PROTO = "E_PROTO"
    E_NODE = "E_NODE"


ALL_CODES = {
    value
    for name, value in vars(ErrorCode).items()
    if name.startswith("E_")
}


class BizError(Exception):
    """Exception carrying a stable business code and structured details."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        if code not in ALL_CODES:
            raise ValueError(f"unknown business code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:
        if self.details:
            return f"{self.code}: {self.message} | {self.details}"
        return f"{self.code}: {self.message}"


EXIT_CODE_FOR_BIZ = {
    ErrorCode.E_FORMAT: 10,
    ErrorCode.E_PORT: 11,
    ErrorCode.E_VERSION: 12,
    ErrorCode.E_CONVERGE: 13,
    ErrorCode.E_IDEMPOTENT: 14,
    ErrorCode.E_REPLAY: 15,
    ErrorCode.E_STATE: 16,
    ErrorCode.E_PROTO: 17,
    ErrorCode.E_NODE: 18,
}
