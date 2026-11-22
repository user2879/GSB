"""Application error types and stable business codes."""

from __future__ import annotations

from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SEED_OUT_OF_RANGE = "E_SEED_OUT_OF_RANGE"
E_NOISE_TABLE_MISMATCH = "E_NOISE_TABLE_MISMATCH"
E_INTERACT_BUDGET_PENDING = "E_INTERACT_BUDGET_PENDING"
E_PATCH_OUT_OF_RANGE = "E_PATCH_OUT_OF_RANGE"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"


class RenderError(Exception):
    """An expected, auditable renderer failure."""

    def __init__(self, code: str, message: str, details: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            out["details"] = self.details
        return out
