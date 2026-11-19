from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    SCENE_INVALID = "E_SCENE_INVALID"
    PATH_DEGENERATE = "E_PATH_DEGENERATE"
    FILL_RULE = "E_FILL_RULE"
    COLOR_RANGE = "E_COLOR_RANGE"
    TOLERANCE_RANGE = "E_TOLERANCE_RANGE"
    LAYER_HASH = "E_LAYER_HASH"
    VERIFIER_MISMATCH = "E_VERIFIER_MISMATCH"
    INJECTION_CONFLICT = "E_INJECTION_CONFLICT"


ERROR_DESCRIPTIONS = {
    ErrorCode.SCENE_INVALID: "场景或坐标超出声明画布边界",
    ErrorCode.PATH_DEGENERATE: "路径包含零长、共线或零宽描边退化",
    ErrorCode.FILL_RULE: "填充规则不是 nonzero 或 evenodd",
    ErrorCode.COLOR_RANGE: "颜色分量超出 0..255",
    ErrorCode.TOLERANCE_RANGE: "扁平化容差必须为有限正数",
    ErrorCode.LAYER_HASH: "分层清单中的层哈希与层内容不一致",
    ErrorCode.VERIFIER_MISMATCH: "合成器与独立验证器结论不一致",
    ErrorCode.INJECTION_CONFLICT: "异常注入目标冲突或无法解析",
}


@dataclass(frozen=True)
class Location:
    layer_id: str | None = None
    path_id: str | None = None
    subpath_index: int | None = None
    segment_index: int | None = None
    field: str | None = None
    pixel: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.layer_id is not None:
            data["layer_id"] = self.layer_id
        if self.path_id is not None:
            data["path_id"] = self.path_id
        if self.subpath_index is not None:
            data["subpath_index"] = self.subpath_index
        if self.segment_index is not None:
            data["segment_index"] = self.segment_index
        if self.field is not None:
            data["field"] = self.field
        if self.pixel is not None:
            data["pixel"] = {"x": self.pixel[0], "y": self.pixel[1]}
        return data


@dataclass
class BusinessError:
    code: ErrorCode
    message: str
    location: Location = field(default_factory=Location)
    injection_id: str | None = None
    recovery: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": str(self.code),
            "description": ERROR_DESCRIPTIONS[self.code],
            "message": self.message,
            "location": self.location.to_dict(),
        }
        if self.injection_id is not None:
            data["injection_id"] = self.injection_id
        if self.recovery is not None:
            data["recovery"] = self.recovery
        if self.details:
            data["details"] = self.details
        return data


def issue_sort_key(item: BusinessError) -> tuple:
    location = item.location
    layer = location.layer_id or "Z"
    y = location.pixel[1] if location.pixel else 10**9
    x = location.pixel[0] if location.pixel else 10**9
    path = location.path_id or ""
    return (
        layer,
        y,
        x,
        path,
        location.subpath_index if location.subpath_index is not None else 10**9,
        location.segment_index if location.segment_index is not None else 10**9,
        location.field or "",
        str(item.code),
    )


def sorted_issues(items: list[BusinessError]) -> list[BusinessError]:
    return sorted(items, key=issue_sort_key)


INJECTION_ERROR_CODES = {
    "coordinate_out_of_bounds": ErrorCode.SCENE_INVALID,
    "degenerate_segment": ErrorCode.PATH_DEGENERATE,
    "fill_rule_invalid": ErrorCode.FILL_RULE,
    "fill_alpha_out_of_range": ErrorCode.COLOR_RANGE,
    "stroke_alpha_out_of_range": ErrorCode.COLOR_RANGE,
    "tolerance_negative": ErrorCode.TOLERANCE_RANGE,
    "stroke_width_zero": ErrorCode.PATH_DEGENERATE,
    "layer_pixel_mutation": ErrorCode.VERIFIER_MISMATCH,
    "layer_hash_mutation": ErrorCode.LAYER_HASH,
}
