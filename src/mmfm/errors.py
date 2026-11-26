"""Business error codes used by the CLI and library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BusinessError(Exception):
    """A deterministic validation/execution failure with a stable code."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    exit_code: int | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)
        if self.exit_code is None:
            if self.code == "E_BUDGET_PENDING":
                self.exit_code = 3
            elif self.code.startswith("E_SNAPSHOT_"):
                self.exit_code = 4
            else:
                self.exit_code = 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def fail(code: str, message: str, **details: Any) -> "None":
    raise BusinessError(code=code, message=message, details=details)


E_USAGE = "E_USAGE"
E_CONFIG_MISSING = "E_CONFIG_MISSING"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_UNKNOWN_TERM = "E_UNKNOWN_TERM"
E_DIMENSION = "E_DIMENSION"
E_PNG_SIGNATURE = "E_PNG_SIGNATURE"
E_PNG_DIMENSION = "E_PNG_DIMENSION"
E_PNG_MODE = "E_PNG_MODE"
E_PNG_CORRUPT = "E_PNG_CORRUPT"
E_MOTION_DIMENSION = "E_MOTION_DIMENSION"
E_MOTION_INVALID = "E_MOTION_INVALID"
E_FRAME_COUNT = "E_FRAME_COUNT"
E_DENOISE_STEPS = "E_DENOISE_STEPS"
E_CHUNK_ROWS = "E_CHUNK_ROWS"
E_FIXED_SCALE = "E_FIXED_SCALE"
E_QUANT_PROFILE = "E_QUANT_PROFILE"
E_ENUM_MISSING = "E_ENUM_MISSING"
E_ENUM_INVALID = "E_ENUM_INVALID"
E_EXCHANGE_VERSION = "E_EXCHANGE_VERSION"
E_V1_WIDTH = "E_V1_WIDTH"
E_V1_PARSE = "E_V1_PARSE"
E_V2_PARSE = "E_V2_PARSE"
E_SIDECAR = "E_SIDECAR"
E_UNKNOWN_COMMAND = "E_UNKNOWN_COMMAND"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_CONDITION_REF = "E_CONDITION_REF"
E_DECODE_OOB = "E_DECODE_OOB"
E_BUDGET_PENDING = "E_BUDGET_PENDING"
E_SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"
E_SNAPSHOT_INCOMPLETE = "E_SNAPSHOT_INCOMPLETE"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_SNAPSHOT_UNCOMMITTED = "E_SNAPSHOT_UNCOMMITTED"
E_TK_UNAVAILABLE = "E_TK_UNAVAILABLE"
