from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


E_CSR_INVALID = "E_CSR_INVALID"
E_MATRIX_NONFINITE = "E_MATRIX_NONFINITE"
E_ACC_WIDTH_INVALID = "E_ACC_WIDTH_INVALID"
E_BLOCK_SIZE_INVALID = "E_BLOCK_SIZE_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_RECOMPUTE_PENDING = "E_RECOMPUTE_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_INCOMPLETE_RESULT = "E_INCOMPLETE_RESULT"
E_BLOCK_COMPUTE_FAILED = "E_BLOCK_COMPUTE_FAILED"


@dataclass(frozen=True)
class Diagnostic:
    code: str
    phase: str
    message: str
    injection_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def sort_key(self) -> tuple[int, str, str, str]:
        injection_order = 0 if self.injection_id is None else 1
        return (injection_order, self.injection_id or "", self.phase, self.code)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "code": self.code,
            "phase": self.phase,
            "message": self.message,
        }
        if self.injection_id is not None:
            value["injection_id"] = self.injection_id
        if self.details:
            value["details"] = self.details
        return value


def sort_diagnostics(items: Iterable[Diagnostic]) -> list[Diagnostic]:
    return sorted(items, key=lambda item: item.sort_key)
