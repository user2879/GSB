"""Failure model shared by loaders, pipeline, scheduler and CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_QUANT_BOUNDS = "E_QUANT_BOUNDS"
E_RECEIPT_ORPHAN = "E_RECEIPT_ORPHAN"
E_RECEIPT_INVALID = "E_RECEIPT_INVALID"
E_WEIGHT_INVALID = "E_WEIGHT_INVALID"
E_WEIGHT_SHAPE = "E_WEIGHT_SHAPE"
E_WEIGHT_DUPLICATE = "E_WEIGHT_DUPLICATE"
E_VERSION_ORDER = "E_VERSION_ORDER"
E_GROUP_INVALID = "E_GROUP_INVALID"
E_CALIBRATION_INVALID = "E_CALIBRATION_INVALID"
E_QUOTA_EXCEEDED = "E_QUOTA_EXCEEDED"
E_BATCH_BUDGET_PENDING = "E_BATCH_BUDGET_PENDING"
E_IO = "E_IO"
E_INJECTION_MISS = "E_INJECTION_MISS"
E_INTERNAL = "E_INTERNAL"


@dataclass
class Failure:
    code: str
    message: str
    stage: str
    layer: str | None = None
    path: str | None = None
    injection_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
        }
        if self.layer is not None:
            data["layer"] = self.layer
        if self.path is not None:
            data["path"] = self.path
        if self.injection_id is not None:
            data["injection_id"] = self.injection_id
        if self.details:
            data["details"] = self.details
        return data


class FailureList:
    def __init__(self) -> None:
        self.items: list[Failure] = []

    def add(
        self,
        code: str,
        message: str,
        stage: str,
        *,
        layer: str | None = None,
        path: str | None = None,
        injection_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> Failure:
        failure = Failure(
            code=code,
            message=message,
            stage=stage,
            layer=layer,
            path=path,
            injection_id=injection_id,
            details=details or {},
        )
        self.items.append(failure)
        return failure

    def extend(self, failures: list[Failure]) -> None:
        self.items.extend(failures)

    def has_code(self, code: str) -> bool:
        return any(item.code == code for item in self.items)

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for item in self.items:
            result[item.code] = result.get(item.code, 0) + 1
        return dict(sorted(result.items()))

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.items]
