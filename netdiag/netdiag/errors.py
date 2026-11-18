"""结构化错误码。

所有诊断错误都携带稳定的机器可读代码、人类可读说明以及定位上下文，
便于账本、日志与 GUI 统一消费。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class DiagError(Exception):
    """全部诊断错误的基类。"""

    code = "E_UNKNOWN"

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context: Dict[str, Any] = dict(context or {})

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}

    def __str__(self) -> str:  # pragma: no cover - 调试便利方法
        return f"{self.code}: {self.message} {self.context}".rstrip()


class WireFormatError(DiagError):
    """线格式抓包样本错误（截断、长度越界、标志位非法、流标识不一致）。"""

    code = "E_WIRE"


class EventFormatError(DiagError):
    """丢包事件清单格式错误（未知版本、字段缺失、负值）。"""

    code = "E_FORMAT"


class MergeConflictError(DiagError):
    """两类来源合并冲突（序号覆盖区间不一致等）。"""

    code = "E_MERGE"


class ReplayMismatchError(DiagError):
    """回环重放或账本历史结论重放不一致。"""

    code = "E_REPLAY"


@dataclass
class ErrorRecord:
    """可序列化的错误记录。"""

    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, exc: DiagError) -> "ErrorRecord":
        return cls(code=exc.code, message=exc.message, context=dict(exc.context))

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}
