"""Business error codes and structured JSON error reporting."""

from __future__ import annotations

import json
from typing import Any


class HapticError(Exception):
    """Error carrying an ``E_xxx`` business code and machine-readable details."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": False, "code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


E_JSON = "E_JSON"
E_FORMAT = "E_FORMAT"
E_DEVICE = "E_DEVICE"
E_EVENT = "E_EVENT"
E_CURVE = "E_CURVE"
E_GLOBAL = "E_GLOBAL"
E_SYNC = "E_SYNC"
E_DIGEST = "E_DIGEST"
E_STATE = "E_STATE"
