"""Stable business error codes and deterministic failure records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


E_CONFIG_JSON = "E_CONFIG_JSON"
E_CONFIG_SCHEMA = "E_CONFIG_SCHEMA"
E_WORKER_DUPLICATE_ID = "E_WORKER_DUPLICATE_ID"
E_WORKER_PORT_RANGE = "E_WORKER_PORT_RANGE"
E_WORKER_PORT_DUPLICATE = "E_WORKER_PORT_DUPLICATE"
E_WORKER_HOST = "E_WORKER_HOST"
E_WORKER_PARAMETER_UNKNOWN = "E_WORKER_PARAMETER_UNKNOWN"
E_MODEL_DIMENSION = "E_MODEL_DIMENSION"
E_QUANTIZATION_BITS = "E_QUANTIZATION_BITS"
E_REQUEST_SEQUENCE = "E_REQUEST_SEQUENCE"
E_REQUEST_ARRIVAL = "E_REQUEST_ARRIVAL"
E_REQUEST_BATCH_SIZE = "E_REQUEST_BATCH_SIZE"
E_REQUEST_WORKER_UNKNOWN = "E_REQUEST_WORKER_UNKNOWN"
E_REQUEST_KIND = "E_REQUEST_KIND"
E_MERGE_WORKER_SET_UNKNOWN = "E_MERGE_WORKER_SET_UNKNOWN"
E_MERGE_PARAMETER_SET_UNKNOWN = "E_MERGE_PARAMETER_SET_UNKNOWN"
E_MERGE_WORKER_UNKNOWN = "E_MERGE_WORKER_UNKNOWN"
E_MERGE_PARAMETER_UNKNOWN = "E_MERGE_PARAMETER_UNKNOWN"
E_LIMIT_INVALID = "E_LIMIT_INVALID"
E_RECEIPT_JSON = "E_RECEIPT_JSON"
E_RECEIPT_ENCODING = "E_RECEIPT_ENCODING"
E_TRUNCATED_RECEIPT = "E_TRUNCATED_RECEIPT"
E_RECEIPT_VERSION = "E_RECEIPT_VERSION"
E_RECEIPT_LAYOUT = "E_RECEIPT_LAYOUT"
E_RECEIPT_FIELD_TYPE = "E_RECEIPT_FIELD_TYPE"
E_UNKNOWN_PARAMETER_VERSION = "E_UNKNOWN_PARAM_VERSION"
E_DURATION_NONFINITE = "E_DURATION_NONFINITE"
E_DUPLICATE_BATCH_ID = "E_DUPLICATE_BATCH_ID"
E_WORKER_QUOTA_EXCEEDED = "E_WORKER_QUOTA_EXCEEDED"
E_NORMALIZE_BUDGET_PENDING = "E_NORMALIZE_BUDGET_PENDING"
E_CONVERSION_TARGET = "E_CONVERSION_TARGET"
E_FRAME_INVALID = "E_FRAME_INVALID"


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    source: str
    ordinal: int = 0
    item: str = ""
    injection_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def sort_key(self) -> tuple[str, int, str, str, str]:
        return (
            self.source,
            int(self.ordinal),
            self.code,
            self.item,
            self.injection_id or "",
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "ordinal": int(self.ordinal),
            "item": self.item,
        }
        if self.injection_id is not None:
            out["injection_id"] = self.injection_id
        if self.details:
            out["details"] = dict(self.details)
        return out


def sort_failures(failures: list[Failure]) -> list[Failure]:
    return sorted(failures, key=Failure.sort_key)
