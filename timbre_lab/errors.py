from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
E_AUDIO_SAMPLE = "E_AUDIO_SAMPLE"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_QUANTIZATION_LEVEL = "E_QUANTIZATION_LEVEL"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_PREVIEW_BUDGET_PENDING = "E_PREVIEW_BUDGET_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_INTERNAL = "E_INTERNAL"


@dataclass(init=False)
class BusinessError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None, **extra: Any) -> None:
        merged = dict(details or {})
        merged.update(extra)
        self.code = code
        self.message = message
        self.details = merged
        super().__init__(code, message, merged)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


def failure(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}
