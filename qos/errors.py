"""业务错误码与结构化异常。

所有可预期的业务失败均以 QosError 子类抛出，携带稳定的业务码，
便于 CLI、GUI 与结构化日志统一处理。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class QosError(Exception):
    code: str = "E_UNKNOWN"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class RuleError(QosError):
    """规则集强校验失败：整体拒绝加载。"""

    code = "E_RULE"


class DependError(QosError):
    """依赖图问题（成环等）。"""

    code = "E_DEPEND"


class RefError(QosError):
    """删除仍被引用的规则。"""

    code = "E_REF"


class IncrementalError(QosError):
    """增量编译与全量编译结果不一致。"""

    code = "E_INCR"


class ShapeError(QosError):
    """整形限速观测越界。"""

    code = "E_SHAPE"


class SnapshotError(QosError):
    """快照版本缺失或快照不可识别。"""

    code = "E_SNAP"


class ReplayError(QosError):
    """快照重放校验出现差异。"""

    code = "E_REPLAY"


@dataclass
class FieldViolation:
    rule_id: str
    field_path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"rule_id": self.rule_id, "field_path": self.field_path, "reason": self.reason}


@dataclass
class RuleLoadResult:
    ok: bool
    violations: list[FieldViolation] = field(default_factory=list)
