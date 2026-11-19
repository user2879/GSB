"""Business error codes and the unified :class:`WorkbenchError` exception.

Every failure that reaches a public entry point is mapped to one of the
explicit codes below instead of raising a generic exception or silently
continuing, so that both the UI and the headless demo can handle the
exceptional path deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass


E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
E_BLOCK_BOUNDS = "E_BLOCK_BOUNDS"
E_COMP_STATE = "E_COMP_STATE"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_BUDGET_EXCEEDED = "E_BUDGET_EXCEEDED"
E_REPORT_MISMATCH = "E_REPORT_MISMATCH"


@dataclass(frozen=True)
class ErrorCode:
    code: str
    message: str


ERROR_CATALOG: dict[str, ErrorCode] = {
    E_AUDIO_FORMAT: ErrorCode(
        E_AUDIO_FORMAT,
        "Illegal sample rate, channel count or PCM bit depth for the workbench.",
    ),
    E_BLOCK_BOUNDS: ErrorCode(
        E_BLOCK_BOUNDS,
        "Block length is outside the allowed bounds or does not align with the stream.",
    ),
    E_COMP_STATE: ErrorCode(
        E_COMP_STATE,
        "Compressor smoothing state is missing or incompatible with the parameters.",
    ),
    E_SNAPSHOT_STALE: ErrorCode(
        E_SNAPSHOT_STALE,
        "Snapshot version roll-back detected; the stale snapshot was rejected.",
    ),
    E_BUDGET_EXCEEDED: ErrorCode(
        E_BUDGET_EXCEEDED,
        "UI refresh budget exceeded persistently; frame updates were degraded.",
    ),
    E_REPORT_MISMATCH: ErrorCode(
        E_REPORT_MISMATCH,
        "Streaming and offline conclusions differ beyond the declared tolerance.",
    ),
}


class WorkbenchError(Exception):
    """Exception carrying one of the declared business error codes."""

    def __init__(self, code: str, detail: str = "") -> None:
        if code not in ERROR_CATALOG:
            raise ValueError(f"Unknown business error code: {code!r}")
        self.code = code
        catalog = ERROR_CATALOG[code]
        self.detail = detail or catalog.message
        super().__init__(f"[{code}] {self.detail}")

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.detail}
