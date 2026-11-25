from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MissingType:
    """Explicit missing-field marker; never coerced to zero."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = MissingType()


def is_missing(value: Any) -> bool:
    return value is MISSING or value == "__MISSING__"


def jsonable(value: Any) -> Any:
    if is_missing(value):
        return "__MISSING__"
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


@dataclass
class ExchangeRecord:
    image_no: int | MissingType = MISSING
    char_no: int | MissingType = MISSING
    bbox: list[float] | list[int] | MissingType = MISSING
    confidence: float | int | MissingType = MISSING
    reading_order: str | int | MissingType = MISSING
    char: str | MissingType = MISSING
    keypoint_version: str | MissingType = MISSING
    keypoint_x: int | MissingType = MISSING
    keypoint_y: int | MissingType = MISSING
    annotation_version: str | MissingType = MISSING
    source_version: str = "unknown"
    present_fields: set[str] = field(default_factory=set)
    missing_fields: set[str] = field(default_factory=set)
    precision_labels: list[str] = field(default_factory=list)
    completeness_labels: list[str] = field(default_factory=list)
    sidecar_id: str | MissingType = MISSING
    residuals: dict[str, float] = field(default_factory=dict)
    source_values: dict[str, Any] = field(default_factory=dict)

    def mark_missing(self, name: str) -> None:
        self.missing_fields.add(name)
        self.completeness_labels.append(f"{name}_missing")

    def to_dict(self) -> dict[str, Any]:
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, set):
                data[key] = sorted(value)
            else:
                data[key] = jsonable(value)
        return data


ORDER_CODE_TO_NAME = {
    10: "rtl_top_to_bottom",
    20: "rtl_top_to_bottom_conflict_resolved",
    30: "manual_authoritative_keypoint",
}
ORDER_NAME_TO_CODE = {name: code for code, name in ORDER_CODE_TO_NAME.items()}
