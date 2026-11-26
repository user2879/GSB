from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_BUDGET_PENDING = "E_BUDGET_PENDING"
E_STALE_SUMMARY = "E_STALE_SUMMARY"
E_PARTIAL_FAILURE = "E_PARTIAL_FAILURE"
E_NONFINITE_LATENT = "E_NONFINITE_LATENT"
E_BLOCK_ROW_OUT_OF_RANGE = "E_BLOCK_ROW_OUT_OF_RANGE"
E_STALE_SNAPSHOT_VERSION = "E_STALE_SNAPSHOT_VERSION"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_CACHE_CORRUPT = "E_CACHE_CORRUPT"
E_INTERNAL = "E_INTERNAL"


@dataclass
class DiagnosticError(Exception):
    code: str
    entry: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)
    injection_id: str | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, f"{self.code} [{self.entry}]: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "entry": self.entry,
            "message": self.message,
        }
        if self.detail:
            result["detail"] = self.detail
        if self.injection_id is not None:
            result["injection_id"] = self.injection_id
        return result


def fail(
    code: str,
    entry: str,
    message: str,
    *,
    detail: dict[str, Any] | None = None,
    injection_id: str | None = None,
) -> None:
    raise DiagnosticError(code, entry, message, detail or {}, injection_id)
