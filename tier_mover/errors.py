"""业务错误码与异常定义。

所有业务异常携带稳定的 code，CLI/GUI 据此映射退出码与结构化日志。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TierMoverError(Exception):
    """所有业务异常的基类。"""

    code = "E_UNKNOWN"

    def __init__(self, message: str, *, field: str | None = None,
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "details": self.details,
        }


class SpecError(TierMoverError):
    """清单强校验失败：整体拒绝加载，不做部分装载。"""

    code = "E_SPEC"


class CapacityError(TierMoverError):
    """层容量不足：列出受影响层与被拒对象。"""

    code = "E_CAPACITY"

    def __init__(self, message: str, *, tier_id: str, rejected_keys: list[str],
                 free_bytes: int, required_bytes: int,
                 field: str | None = None,
                 details: dict[str, Any] | None = None):
        super().__init__(message, field=field, details=details)
        self.tier_id = tier_id
        self.rejected_keys = rejected_keys
        self.free_bytes = free_bytes
        self.required_bytes = required_bytes
        self.details.update({
            "tier_id": tier_id,
            "rejected_keys": rejected_keys,
            "free_bytes": free_bytes,
            "required_bytes": required_bytes,
        })


class IntegrityError(TierMoverError):
    """对象文件/目录字节数/摘要完整性被破坏。"""

    code = "E_INTEGRITY"


class VersionConflictError(TierMoverError):
    """账本格式版本或策略版本冲突。"""

    code = "E_VERSION"


class ReplayError(TierMoverError):
    """恢复后重放校验在首个分歧点失败。"""

    code = "E_REPLAY"

    def __init__(self, message: str, *, object_key: str | None,
                 mismatch_field: str, basis: str,
                 expected: Any, actual: Any,
                 details: dict[str, Any] | None = None):
        super().__init__(message, field=mismatch_field, details=details)
        self.object_key = object_key
        self.mismatch_field = mismatch_field
        self.basis = basis
        self.expected = expected
        self.actual = actual
        self.details.update({
            "object_key": object_key,
            "mismatch_field": mismatch_field,
            "basis": basis,
            "expected": expected,
            "actual": actual,
        })


@dataclass
class ErrorCollector:
    """收集校验违规；first 为首个违规字段与原因（按校验顺序稳定）。"""

    violations: list[dict[str, str]] = field(default_factory=list)

    def add(self, field_path: str, reason: str) -> None:
        self.violations.append({"field": field_path, "reason": reason})

    def check(self) -> None:
        if self.violations:
            first = self.violations[0]
            raise SpecError(
                f"清单校验失败，首个违规字段 {first['field']}: {first['reason']} "
                f"(共 {len(self.violations)} 处违规)",
                field=first["field"],
                details={"all_violations": self.violations},
            )
