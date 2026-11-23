"""业务错误码、异常与失败清单记录。"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---- 业务错误码（需求条目 1/5/6/7/8） ----
E_FIELD_TRUNCATED = "E_FIELD_TRUNCATED"
E_DUPLICATE_VIEW = "E_DUPLICATE_VIEW"
E_COORD_OUT_OF_RANGE = "E_COORD_OUT_OF_RANGE"
E_SCORE_OUT_OF_RANGE = "E_SCORE_OUT_OF_RANGE"
E_VOXEL_NOT_ALIGNED = "E_VOXEL_NOT_ALIGNED"
E_VOXEL_ZERO_SIZE = "E_VOXEL_ZERO_SIZE"
E_NON_FINITE = "E_NON_FINITE"
E_UNKNOWN_METRIC = "E_UNKNOWN_METRIC"
E_PARSE = "E_PARSE"
E_TRUNCATED_PNG = "E_TRUNCATED_PNG"
E_MISSING_CAMERA = "E_MISSING_CAMERA"
E_DISPARITY_OUT_OF_RANGE = "E_DISPARITY_OUT_OF_RANGE"
E_VIEW_BUDGET_EXHAUSTED = "E_VIEW_BUDGET_EXHAUSTED"
E_POINT_BUDGET_EXHAUSTED = "E_POINT_BUDGET_EXHAUSTED"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_MATCH_INLIERS = "E_MATCH_INLIERS"

# 异常注入故障类型 -> 业务码
INJECTION_CODES = {
    "truncated_png": E_TRUNCATED_PNG,
    "missing_camera": E_MISSING_CAMERA,
    "disparity_overflow": E_DISPARITY_OUT_OF_RANGE,
    "non_finite_feature": E_NON_FINITE,
    "zero_voxel_size": E_VOXEL_ZERO_SIZE,
}


class MVRBError(Exception):
    """工作台基础异常，携带业务错误码。"""

    def __init__(self, code: str, message: str = "", context: dict | None = None):
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.context = context or {}


@dataclass
class Failure:
    """失败清单条目；顺序由 sort_key 稳定决定。"""

    code: str
    message: str
    view: int | None = None
    stage: str | None = None
    sort_key: str = ""
    injection_id: str | None = None
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "view": self.view,
            "stage": self.stage,
            "sort_key": self.sort_key,
            "injection_id": self.injection_id,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Failure":
        return cls(**d)


class FailureList:
    """可累积、按稳定排序键输出的失败清单。"""

    def __init__(self) -> None:
        self._items: list[Failure] = []

    def add(self, failure: Failure) -> None:
        self._items.append(failure)

    def fail(
        self,
        code: str,
        message: str,
        view: int | None = None,
        stage: str | None = None,
        sort_key: str | None = None,
        injection_id: str | None = None,
        context: dict | None = None,
    ) -> Failure:
        key = sort_key or f"{stage or '':>8}|{view if view is not None else -1:04d}|{code}"
        f = Failure(code, message, view, stage, key, injection_id, context or {})
        self._items.append(f)
        return f

    def sorted(self) -> list[Failure]:
        return sorted(self._items, key=lambda f: (f.sort_key, f.code, f.view or -1))

    def dicts(self) -> list[dict]:
        return [f.to_dict() for f in self.sorted()]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)
