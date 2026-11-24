"""Stable business error codes and issue helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_TEXTURE_HEADER = "E_TEXTURE_HEADER"
E_TEXTURE_SIZE = "E_TEXTURE_SIZE"
E_TEXTURE_MODE = "E_TEXTURE_MODE"
E_TEXTURE_DECODE = "E_TEXTURE_DECODE"
E_TEXTURE_MISSING = "E_TEXTURE_MISSING"
E_CONFIG_JSON = "E_CONFIG_JSON"
E_MATERIALS_JSON = "E_MATERIALS_JSON"
E_COMMANDS_JSON = "E_COMMANDS_JSON"
E_SCHEMA = "E_SCHEMA"
E_RESOLUTION_TIERS = "E_RESOLUTION_TIERS"
E_RESOLUTION_TIER = "E_RESOLUTION_TIER"
E_NOISE_PARAMS = "E_NOISE_PARAMS"
E_QUANTIZATION = "E_QUANTIZATION"
E_ROUNDING_MODE = "E_ROUNDING_MODE"
E_CONCURRENCY = "E_CONCURRENCY"
E_BUDGET = "E_BUDGET"
E_LATENCY = "E_LATENCY"
E_MATERIAL_WEIGHTS = "E_MATERIAL_WEIGHTS"
E_LAYER_REFERENCE = "E_LAYER_REFERENCE"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_RESPONSE_STALE = "E_RESPONSE_STALE"
E_LAYER_RUNTIME = "E_LAYER_RUNTIME"
E_MATERIAL_RUNTIME = "E_MATERIAL_RUNTIME"
E_SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"
E_SNAPSHOT_IO = "E_SNAPSHOT_IO"
E_INJECTION_CONFIG = "E_INJECTION_CONFIG"
E_DIAGNOSTIC_IO = "E_DIAGNOSTIC_IO"
E_INTERNAL = "E_INTERNAL"

ALL_ERROR_CODES = frozenset(
    value
    for name, value in globals().items()
    if name.startswith("E_") and isinstance(value, str)
)


@dataclass
class Issue:
    """A single isolated input, runtime, or reporting problem."""

    code: str
    message: str
    source: str = ""
    item: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "item": self.item,
            "details": self.details,
        }


class InjectedFailure(RuntimeError):
    """Raised by an explicit anomaly injection."""

    def __init__(self, code: str, message: str, injection_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.injection_id = injection_id


def make_issue(
    code: str,
    message: str,
    *,
    source: str = "",
    item: str = "",
    details: dict[str, Any] | None = None,
) -> Issue:
    return Issue(code=code, message=message, source=source, item=item, details=details or {})
