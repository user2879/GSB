from __future__ import annotations

from typing import Any


class CloudShardError(Exception):
    code = "E_INTERNAL"

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        field: str | None = None,
        basis: str | dict[str, Any] | None = None,
        shard_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.field = field
        self.basis = basis
        self.shard_id = shard_id

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "error": self.message,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.field is not None:
            payload["field"] = self.field
        if self.shard_id is not None:
            payload["shard_id"] = self.shard_id
        if self.basis is not None:
            payload["basis"] = self.basis
        return payload


class SpecError(CloudShardError):
    code = "E_SPEC"


class PortInUseError(CloudShardError):
    code = "E_PORT"


class ProtocolTimeoutError(CloudShardError):
    code = "E_TIMEOUT"


class FieldError(CloudShardError):
    code = "E_FIELD"


class LeaseError(CloudShardError):
    code = "E_LEASE"


class VerifyError(CloudShardError):
    code = "E_VERIFY"


class EnvironmentError(CloudShardError):
    code = "E_ENV"


ERROR_EXIT_CODES = {
    SpecError: 2,
    PortInUseError: 3,
    ProtocolTimeoutError: 4,
    FieldError: 5,
    LeaseError: 6,
    VerifyError: 7,
    EnvironmentError: 8,
}


def exit_code_for(exc: BaseException) -> int:
    for cls, code in ERROR_EXIT_CODES.items():
        if isinstance(exc, cls):
            return code
    return 1
