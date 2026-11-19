"""业务错误码与工作台异常定义。

所有错误码均以 ``E_`` 开头，可被 CLI 序列化为 JSON 诊断报告，
也可被失败隔离逻辑捕获并写入 failure 清单。
"""
from __future__ import annotations

from typing import Any, Mapping, Optional


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SEED_OUT_OF_RANGE = "E_SEED_OUT_OF_RANGE"
E_GUIDANCE_INVALID = "E_GUIDANCE_INVALID"
E_STEP_ZERO = "E_STEP_ZERO"
E_NOISE_TABLE_MISMATCH = "E_NOISE_TABLE_MISMATCH"
E_INIT_IMAGE_INVALID = "E_INIT_IMAGE_INVALID"
E_PATCH_OUT_OF_RANGE = "E_PATCH_OUT_OF_RANGE"
E_STATE_DIVERGENCE = "E_STATE_DIVERGENCE"
E_STAGE_FAILED = "E_STAGE_FAILED"
E_INJECTION = "E_INJECTION"


class WorkbenchError(Exception):
    """带业务码的工作台异常，可携带阶段/步/帧定位信息。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: Optional[str] = None,
        step: Optional[int] = None,
        frame: Optional[int] = None,
        injection_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.step = step
        self.frame = frame
        self.injection_id = injection_id

    def to_record(self, exception_type: Optional[str] = None) -> dict[str, Any]:
        record = {
            "code": self.code,
            "message": self.message,
            "exception_type": exception_type or type(self).__name__,
            "stage": self.stage,
            "step": self.step,
            "frame": self.frame,
            "injection_id": self.injection_id,
        }
        return {key: value for key, value in record.items() if value is not None}

    def __str__(self) -> str:
        location = self.stage or "workbench"
        if self.step is not None:
            location = f"{location}#step{self.step}"
        if self.frame is not None:
            location = f"{location}#frame{self.frame}"
        return f"[{self.code}] {location}: {self.message}"


def fail_record(
    code: str,
    message: str,
    *,
    stage: Optional[str] = None,
    step: Optional[int] = None,
    frame: Optional[int] = None,
    exception_type: Optional[str] = "WorkbenchError",
    injection_id: Optional[str] = None,
) -> dict[str, Any]:
    err = WorkbenchError(
        code,
        message,
        stage=stage,
        step=step,
        frame=frame,
        injection_id=injection_id,
    )
    record = err.to_record(exception_type)
    return record


def ensure_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkbenchError(
            E_CONFIG_INVALID, f"字段 {field} 必须是对象/映射，实际为 {type(value).__name__}"
        )
    return value
