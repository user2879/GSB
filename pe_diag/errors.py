"""Business error codes and the :class:`DiagError` exception.

All recoverable, per-sample/per-stage failures raise :class:`DiagError` with a
stable code.  The orchestrator catches them, records them in the failure list
and keeps processing the remaining tasks, so a single fault never aborts a run.
"""

from __future__ import annotations

# --- business error codes (stable strings, part of the public contract) -----
E_NONFINITE_INPUT = "E_NONFINITE_INPUT"
E_KEY_POS_OOR = "E_KEY_POS_OOR"
E_BASE_ZERO = "E_BASE_ZERO"
E_SCALE_INVALID = "E_SCALE_INVALID"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_DIM_MISMATCH = "E_DIM_MISMATCH"
E_CONFIG = "E_CONFIG"

ALL_CODES = (
    E_NONFINITE_INPUT,
    E_KEY_POS_OOR,
    E_BASE_ZERO,
    E_SCALE_INVALID,
    E_SNAPSHOT_STALE,
    E_DIM_MISMATCH,
    E_CONFIG,
)


class DiagError(Exception):
    """A deterministic, code-carrying failure of one task."""

    def __init__(self, code: str, message: str, *, injection_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.injection_id = injection_id

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "injection_id": self.injection_id,
        }
