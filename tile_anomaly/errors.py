"""Business error codes and records used across pipeline stages."""

from dataclasses import dataclass, field
from typing import Any


E_TILE_FORMAT = "E_TILE_FORMAT"
E_TILE_QUOTA_EXCEEDED = "E_TILE_QUOTA_EXCEEDED"
E_BUDGET_PENDING = "E_BUDGET_PENDING"
E_FUSION_PENDING = "E_FUSION_PENDING"
E_CONFIG = "E_CONFIG"
E_WINDOW_CONTEXT_UNAVAILABLE = "E_WINDOW_CONTEXT_UNAVAILABLE"
E_WINDOW_FEATURE = "E_WINDOW_FEATURE"
E_IO = "E_IO"


@dataclass(frozen=True)
class BusinessEvent:
    tile_id: str | None
    stage: str
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }


class ConfigurationError(ValueError):
    def __init__(self, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = E_CONFIG
        self.detail = detail or {}


class StageFailure(RuntimeError):
    def __init__(self, code: str, message: str, detail: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.detail = detail or {}

