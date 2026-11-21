"""业务错误码与可诊断异常。

所有错误码均以 ``E_`` 开头，进入失败清单与诊断报告，保证可追溯。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class BizCode:
    E_VIEW_INVALID = "E_VIEW_INVALID"
    E_CAMERA_MISSING = "E_CAMERA_MISSING"
    E_PNG_TRUNCATED = "E_PNG_TRUNCATED"
    E_RESOLUTION_MISMATCH = "E_RESOLUTION_MISMATCH"
    E_VIEW_DUPLICATE = "E_VIEW_DUPLICATE"
    E_INTRINSIC_SINGULAR = "E_INTRINSIC_SINGULAR"
    E_FOV_ILLEGAL = "E_FOV_ILLEGAL"
    E_FEATURE_NONFINITE = "E_FEATURE_NONFINITE"
    E_MIN_INLIERS = "E_MIN_INLIERS"
    E_MIN_DISPARITY = "E_MIN_DISPARITY"
    E_VOXEL_ZERO = "E_VOXEL_ZERO"
    E_CONFIG_INVALID = "E_CONFIG_INVALID"
    E_SNAPSHOT_MISMATCH = "E_SNAPSHOT_MISMATCH"
    E_REBUILD_BUDGET_PENDING = "E_REBUILD_BUDGET_PENDING"
    E_INJECTION = "E_INJECTION"


@dataclass
class FailureRecord:
    """单条失败记录（视图级/任务级异常隔离的最小单元）。"""

    view: Optional[int]
    stage: str
    code: str
    reason: str
    injection_id: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "view": self.view,
            "stage": self.stage,
            "code": self.code,
            "reason": self.reason,
            "injection_id": self.injection_id,
            "detail": self.detail,
        }


class WorkbenchError(Exception):
    """带业务码的可诊断异常，默认可隔离处理。"""

    code = "E_INJECTION"

    def __init__(
        self,
        reason: str,
        *,
        view: Optional[int] = None,
        stage: str = "",
        code: Optional[str] = None,
        injection_id: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.view = view
        self.stage = stage
        if code is not None:
            self.code = code
        self.injection_id = injection_id
        self.detail = detail or {}

    def to_record(self) -> FailureRecord:
        return FailureRecord(
            view=self.view,
            stage=self.stage,
            code=self.code,
            reason=self.reason,
            injection_id=self.injection_id,
            detail=self.detail,
        )
