"""Business error codes and process exit-code mapping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EXIT_OK = 0
EXIT_VALIDATION = 2
EXIT_RENDER = 3
EXIT_STATE = 4
EXIT_INTERNAL = 5


@dataclass
class BizError(Exception):
    """A deterministic business error suitable for JSON reports."""

    code: str
    message: str
    stage: str | None = None
    layer: int | None = None
    channel: str | None = None
    path: str | None = None
    injected: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "injected": self.injected,
        }
        for key in ("stage", "layer", "channel", "path"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.details:
            result["details"] = self.details
        return result


def issue(
    code: str,
    message: str,
    *,
    stage: str | None = None,
    layer: int | None = None,
    channel: str | None = None,
    path: str | None = None,
    injected: bool = False,
    **details: Any,
) -> dict[str, Any]:
    return BizError(
        code=code,
        message=message,
        stage=stage,
        layer=layer,
        channel=channel,
        path=path,
        injected=injected,
        details=details,
    ).to_dict()


VALIDATION_CODES = {
    "E_ARG",
    "E_IO",
    "E_JSON",
    "E_CONFIG",
    "E_RES_TIER",
    "E_RESOLUTION",
    "E_FREQ",
    "E_AMPLITUDE",
    "E_LAYER_COUNT",
    "E_LAYER_BUDGET",
    "E_BATCH_BUDGET",
    "E_FRAC_BITS",
    "E_QUANT_STEP",
    "E_TIE_RULE",
    "E_BLEND_MODE",
    "E_PREVIEW_BUDGET",
    "E_THREAD_POOL",
    "E_COMMIT_INTERVAL",
    "E_TOLERANCE",
    "E_PNG_HEADER",
    "E_PNG_RESOLUTION",
    "E_PNG_MODE",
    "E_PNG_DECODE",
    "E_MATERIAL",
    "E_MATERIAL_CHANNEL",
    "E_WEIGHT",
    "E_COMMAND",
    "E_INJECTION",
}

RENDER_CODES = {
    "E_POINT_BUDGET_PENDING",
    "E_CHANNEL_DEPENDENCY",
    "E_RENDER",
    "E_INJECTED",
    "E_PREVIEW_STALE",
    "E_TEXTURE_UNAVAILABLE",
}

STATE_CODES = {
    "E_SNAPSHOT_CORRUPT",
    "E_SNAPSHOT_VERSION",
    "E_SNAPSHOT_IO",
    "E_CONFIG_CHANGED_REBUILD",
    "E_TEXTURE_EXTERNAL_CHANGE",
}


def exit_code_for_codes(codes: set[str] | list[str]) -> int:
    values = set(codes)
    if values & STATE_CODES:
        return EXIT_STATE
    if values & RENDER_CODES:
        return EXIT_RENDER
    if values & VALIDATION_CODES:
        return EXIT_VALIDATION
    return EXIT_OK
