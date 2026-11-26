"""业务码与带样本标识/位置的校验异常。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_SAMPLE_INVALID = "E_SAMPLE_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_KEYPOINT_VERSION_INVALID = "E_KEYPOINT_VERSION_INVALID"
E_SCAN_BUDGET_EXHAUSTED = "E_SCAN_BUDGET_EXHAUSTED"
E_COMPARE_BUDGET_EXHAUSTED = "E_COMPARE_BUDGET_EXHAUSTED"
E_ADJUDICATION_BUDGET = "E_ADJUDICATION_BUDGET"
E_SNAPSHOT_VERSION_MISMATCH = "E_SNAPSHOT_VERSION_MISMATCH"
E_RECOVERY_FALLBACK = "E_RECOVERY_FALLBACK"


@dataclass
class ValidationIssue:
    code: str
    sample_id: str
    location: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "sample_id": self.sample_id,
            "location": self.location,
            "message": self.message,
            "payload": self.payload,
        }


class PipelineValidationError(ValueError):
    """单个样本/配置不合规时抛出，由加载层捕获并隔离。"""

    def __init__(self, code: str, sample_id: str, location: str, message: str,
                 payload: dict[str, Any] | None = None) -> None:
        super().__init__(f"[{code}] {sample_id} {location}: {message}")
        self.issue = ValidationIssue(code, sample_id, location, message, payload or {})


class BudgetExhaustedError(RuntimeError):
    """某类预算不足，调度器据此把单元转入待办而不是失败。"""

    def __init__(self, budget_name: str, required: int, remaining: int) -> None:
        super().__init__(f"budget {budget_name} exhausted: need {required}, remain {remaining}")
        self.budget_name = budget_name
        self.required = required
        self.remaining = remaining
