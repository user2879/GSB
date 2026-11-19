"""结构化业务错误码与异常链路。

所有可预期的输入/状态问题都抛出 :class:`TacticalError` 的子类，
携带稳定错误码、上下文与可选的关联依赖边，便于上层生成结构化诊断，
且保证持久化的已提交状态不会被异常写坏（写入均走原子提交）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


class ErrorCode:
    E_MAP_SHAPE = "E_MAP_SHAPE"
    E_TERRAIN_RANGE = "E_TERRAIN_RANGE"
    E_DEPENDENCY_STALE = "E_DEPENDENCY_STALE"
    E_CLOSURE_MISMATCH = "E_CLOSURE_MISMATCH"
    E_BUDGET_DENIED = "E_BUDGET_DENIED"
    E_UNIT_INVALID = "E_UNIT_INVALID"
    E_QUERY_UNREACHABLE = "E_QUERY_UNREACHABLE"


@dataclass
class TacticalError(Exception):
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    dependency_edge: Optional[str] = None

    def __post_init__(self) -> None:
        super().__init__(f"{self.code}: {self.message}")

    def to_diagnostic(self) -> Dict[str, Any]:
        diag: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }
        if self.dependency_edge is not None:
            diag["dependency_edge"] = self.dependency_edge
        return diag


class MapShapeError(TacticalError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.E_MAP_SHAPE, message, context)


class TerrainRangeError(TacticalError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.E_TERRAIN_RANGE, message, context)


class DependencyStaleError(TacticalError):
    """缺失节点 / 内容哈希不匹配 / 引用悬空。"""

    def __init__(self, message: str, dependency_edge: Optional[str] = None, **context: Any) -> None:
        super().__init__(ErrorCode.E_DEPENDENCY_STALE, message, context, dependency_edge)


class ClosureMismatchError(TacticalError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.E_CLOSURE_MISMATCH, message, context)


class BudgetDeniedError(TacticalError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.E_BUDGET_DENIED, message, context)


class UnitInvalidError(TacticalError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.E_UNIT_INVALID, message, context)


class QueryUnreachableError(TacticalError):
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(ErrorCode.E_QUERY_UNREACHABLE, message, context)


def diagnostic_dict(exc: TacticalError) -> Dict[str, Any]:
    return asdict(exc) if False else exc.to_diagnostic()
