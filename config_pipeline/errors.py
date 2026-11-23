"""Typed business errors used by the local distribution pipeline."""

from __future__ import annotations

from typing import Any, Mapping


E_MANIFEST = "E_MANIFEST"
E_PORT = "E_PORT"
E_HANDSHAKE = "E_HANDSHAKE"
E_REPLY = "E_REPLY"
E_DIVERGE = "E_DIVERGE"
E_REPLAY = "E_REPLAY"


class PipelineError(Exception):
    """An error with a stable business code and machine-readable location."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        shard_id: str | None = None,
        details: Mapping[str, Any] | None = None,
        basis: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.shard_id = shard_id
        self.details = dict(details or {})
        self.basis = list(basis or ())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.field is not None:
            payload["field"] = self.field
        if self.shard_id is not None:
            payload["shard_id"] = self.shard_id
        if self.details:
            payload["details"] = self.details
        if self.basis:
            payload["basis"] = self.basis
        return payload

    def __str__(self) -> str:
        location = ""
        if self.field:
            location += f" field={self.field}"
        if self.shard_id:
            location += f" shard_id={self.shard_id}"
        return f"{self.code}{location}: {self.message}"
