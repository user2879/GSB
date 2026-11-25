"""Business error codes and structured validation issues."""

from __future__ import annotations

from dataclasses import dataclass


E_LAYER_UNKNOWN = "E_LAYER_UNKNOWN"
E_LAYER_DUPLICATE = "E_LAYER_DUPLICATE"
E_GROUP_GRANULARITY = "E_GROUP_GRANULARITY"
E_WEIGHT_NONFINITE = "E_WEIGHT_NONFINITE"
E_SHAPE_INVALID = "E_SHAPE_INVALID"
E_VERSION_ORDER = "E_VERSION_ORDER"
E_RECEIPT_REFERENCE = "E_RECEIPT_REFERENCE"
E_RECEIPT_FORMAT = "E_RECEIPT_FORMAT"
E_TIME_MS_NEGATIVE = "E_TIME_MS_NEGATIVE"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_QUOTA_EXCEEDED = "E_QUOTA_EXCEEDED"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"


@dataclass(frozen=True)
class Issue:
    """A single structured loading / validation / scheduling issue."""

    code: str
    message: str
    layer: str | None = None
    detail: dict | None = None

    def to_dict(self) -> dict:
        out = {"code": self.code, "message": self.message}
        if self.layer is not None:
            out["layer"] = self.layer
        if self.detail:
            out["detail"] = self.detail
        return out


class QuantCalibError(Exception):
    """Raised for fatal, non-recoverable errors."""

    def __init__(self, code: str, message: str, layer: str | None = None):
        super().__init__(f"{code}: {message}" + (f" (layer={layer})" if layer else ""))
        self.code = code
        self.layer = layer
