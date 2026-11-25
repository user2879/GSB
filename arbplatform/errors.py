"""Business error codes and exception types.

Each failure maps to a stable business code required by the specification:

* ``E_SPEC``    - declaration / manifest validation failure
* ``E_PORT``    - loopback port occupied or bind/connect failure
* ``E_TIMEOUT`` - real socket handshake / IO timeout
* ``E_REPLY``   - service reply missing required fields
* ``E_REPLAY``  - ledger replay mismatch after forced termination
* ``E_FIXED``   - fixed-point quantization rule violation
* ``E_STATE``   - invalid platform state transition
"""
from __future__ import annotations

from typing import Any


class ArbError(Exception):
    """Base class carrying a stable business code and a located field."""

    code = "E_STATE"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        code: str | None = None,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        if code is not None:
            self.code = code
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            out["field"] = self.field
        if self.detail is not None:
            out["detail"] = self.detail
        return out


class SpecError(ArbError):
    code = "E_SPEC"


class PortError(ArbError):
    code = "E_PORT"


class TimeoutError(ArbError):  # noqa: A001 - intentional domain name
    code = "E_TIMEOUT"


class ReplyError(ArbError):
    code = "E_REPLY"


class ReplayError(ArbError):
    code = "E_REPLAY"


class FixedPointError(ArbError):
    code = "E_FIXED"


class StateError(ArbError):
    code = "E_STATE"
