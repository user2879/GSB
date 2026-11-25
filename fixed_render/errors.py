from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CODES = {
    "INPUT_INVALID": "E_INPUT_INVALID",
    "LEASE_EXPIRED": "E_LEASE_EXPIRED",
    "LEASE_STALE": "E_LEASE_STALE",
    "LEASE_BUSY": "E_LEASE_BUSY",
    "DIM_MISMATCH": "E_DIM_MISMATCH",
    "NONFINITE_LATENT": "E_NONFINITE_LATENT",
    "TILE_OUT_OF_RANGE": "E_TILE_OUT_OF_RANGE",
    "QUANT_INVALID": "E_QUANT_INVALID",
    "PREVIEW_STALE": "E_PREVIEW_STALE",
    "STAGE_FAILED": "E_STAGE_FAILED",
    "INJECTED": "E_INJECTED_FAULT",
}


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    field_name: str | None = None
    fields: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        where = f" [{self.field_name}]" if self.field_name else ""
        return f"{self.code}{where}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "field": self.field_name,
            "fields": self.fields,
            "details": self.details,
        }


class InputInvalid(BusinessError):
    def __init__(
        self,
        message: str = "input validation failed",
        field_name: str | None = None,
        fields: dict[str, str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        normalized = fields or {}
        if field_name and field_name not in normalized:
            normalized[field_name] = message
        super().__init__(
            CODES["INPUT_INVALID"],
            message,
            field_name,
            normalized,
            details or {},
        )


class LeaseExpired(BusinessError):
    def __init__(self, message: str = "lease has expired and slot was reclaimed", **details: Any) -> None:
        super().__init__(CODES["LEASE_EXPIRED"], message, details=details)


class LeaseStale(BusinessError):
    def __init__(self, message: str = "lease generation is stale", **details: Any) -> None:
        super().__init__(CODES["LEASE_STALE"], message, details=details)


class StageFault(BusinessError):
    pass
