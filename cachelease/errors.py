from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheLeaseError(Exception):
    code: str
    message: str
    location: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.location is not None:
            error["location"] = self.location
        if self.details:
            error["details"] = self.details
        return {"error": error}


class SpecError(CacheLeaseError):
    def __init__(self, message: str, location: str) -> None:
        super().__init__("E_SPEC", message, location)


class HandshakeError(CacheLeaseError):
    def __init__(self, message: str, location: str | None = None) -> None:
        super().__init__("E_HANDSHAKE", message, location)


class ReplyError(CacheLeaseError):
    def __init__(
        self, message: str, location: str | None = None, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__("E_REPLY", message, location, details or {})


class ProtocolError(CacheLeaseError):
    def __init__(
        self, code: str, message: str, location: str | None = None, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(code, message, location, details or {})


class DivergenceError(CacheLeaseError):
    def __init__(self, node_id: str, artifact_key: str, field_name: str, detail: str) -> None:
        super().__init__(
            "E_DIVERGE",
            f"incremental and full projections differ at {node_id}/{artifact_key}/{field_name}: {detail}",
            f"{node_id}/{artifact_key}/{field_name}",
            {"node_id": node_id, "artifact_key": artifact_key, "field": field_name},
        )


class ReplayError(CacheLeaseError):
    def __init__(self, seq: int, detail: str) -> None:
        super().__init__(
            "E_REPLAY",
            f"ledger replay mismatch at event seq {seq}: {detail}",
            f"event[{seq}]",
            {"seq": seq},
        )

