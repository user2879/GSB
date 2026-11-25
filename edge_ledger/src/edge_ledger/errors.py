from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EdgeLedgerError(Exception):
    code: str
    message: str
    field: str | None = None
    line: int | None = None
    shard_id: str | None = None
    basis: str | None = None

    def __str__(self) -> str:
        parts = [self.code, self.message]
        if self.field is not None:
            parts.append(f"field={self.field}")
        if self.line is not None:
            parts.append(f"line={self.line}")
        if self.shard_id is not None:
            parts.append(f"shard={self.shard_id}")
        if self.basis is not None:
            parts.append(f"basis={self.basis}")
        return ": ".join(parts)


class SpecError(EdgeLedgerError):
    def __init__(self, message: str, field: str, line: int = 1):
        super().__init__("E_SPEC", message, field=field, line=line)


class FormatError(EdgeLedgerError):
    def __init__(self, message: str, field: str, line: int = 1):
        super().__init__("E_FORMAT", message, field=field, line=line)


class LeaseError(EdgeLedgerError):
    def __init__(self, message: str, field: str | None = "lease_version", **kwargs):
        super().__init__("E_LEASE", message, field=field, **kwargs)


class QuotaError(EdgeLedgerError):
    def __init__(self, message: str, field: str | None = "quota", **kwargs):
        super().__init__("E_QUOTA", message, field=field, **kwargs)


class ReplayError(EdgeLedgerError):
    def __init__(
        self,
        message: str,
        field: str,
        shard_id: str | None = None,
        basis: str | None = None,
    ):
        super().__init__(
            "E_REPLAY",
            message,
            field=field,
            shard_id=shard_id,
            basis=basis,
        )
