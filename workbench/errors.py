from __future__ import annotations

from typing import Any


class WorkbenchError(Exception):
    """Business error with a stable machine-readable code."""

    exit_code = 2

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "error": {"code": self.code, "message": self.message},
        }
        if self.details:
            payload["error"]["details"] = _jsonable(self.details)
        return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
