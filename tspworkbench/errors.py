"""业务错误码与工作台异常类型。

所有可恢复的输入/预算/布局故障都以 :class:`WorkbenchError` 携带稳定业务码
进入失败清单；真正的编程缺陷仍以普通异常向上抛出。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---- 稳定业务码（验收要求注入项必须映射到这些码） ----
E_FIELD_LAYOUT = "E_FIELD_LAYOUT"
E_DUPLICATE_KEY = "E_DUPLICATE_KEY"
E_LAG_NEGATIVE = "E_LAG_NEGATIVE"
E_MODEL_ENUM = "E_MODEL_ENUM"
E_MISSING_ENUM = "E_MISSING_ENUM"
E_SERIES_BUDGET_EXHAUSTED = "E_SERIES_BUDGET_EXHAUSTED"
E_GLOBAL_BUDGET_EXHAUSTED = "E_GLOBAL_BUDGET_EXHAUSTED"
E_TIMESTAMP_IRREGULAR = "E_TIMESTAMP_IRREGULAR"
E_NON_FINITE = "E_NON_FINITE"
E_WINDOW_TOO_LONG = "E_WINDOW_TOO_LONG"
E_QUANT_GEAR_INVALID = "E_QUANT_GEAR_INVALID"
E_PARSE = "E_PARSE"
E_SIDECAR_MISMATCH = "E_SIDECAR_MISMATCH"


@dataclass
class WorkbenchError(Exception):
    """携带业务码与定位信息的受控异常。"""

    code: str
    message: str
    injection_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - 调试便利
        loc = f" [injection={self.injection_id}]" if self.injection_id else ""
        return f"{self.code}: {self.message}{loc}"

    def to_record(self) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }
        if self.injection_id is not None:
            rec["injection_id"] = self.injection_id
        return rec
