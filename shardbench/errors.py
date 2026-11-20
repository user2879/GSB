"""结构化业务异常与业务码定义。

所有可预期的领域错误都抛出 :class:`ShardBenchError` 的子类，携带稳定业务码、
人类可读信息以及出错字段路径（如适用），便于结构化日志与界面统一呈现。
"""

from __future__ import annotations

from typing import Any, Iterable, Optional


class ShardBenchError(Exception):
    """工作台领域异常基类。"""

    code = "E_UNKNOWN"

    def __init__(
        self,
        message: str,
        *,
        paths: Optional[Iterable[str]] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.paths: list[str] = list(paths) if paths else []
        self.details: dict[str, Any] = dict(details) if details else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "paths": list(self.paths),
            "details": self.details,
        }


class ConfigError(ShardBenchError):
    """配置非法，整体拒绝加载。"""

    code = "E_CONFIG"


class CycleError(ShardBenchError):
    """路由单元依赖图存在环。"""

    code = "E_CYCLE"

    def __init__(self, cycle: Iterable[str]) -> None:
        self.cycle: list[str] = list(cycle)
        super().__init__(
            "路由单元依赖图存在环",
            paths=[f"dependencies.{name}" for name in self.cycle],
            details={"cycle": self.cycle},
        )


class RingError(ShardBenchError):
    """一致性哈希环非法（例如同权重节点冲突）。"""

    code = "E_RING"


class VersionError(ShardBenchError):
    """快照/磁盘元数据版本不一致。"""

    code = "E_VERSION"


class ReplayError(ShardBenchError):
    """账本重放结果与已提交结论不一致。"""

    code = "E_REPLAY"

    def __init__(
        self,
        message: str,
        *,
        unit: Optional[str] = None,
        target: Optional[str] = None,
        field: Optional[str] = None,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        details = {
            "unit": unit,
            "target": target,
            "field": field,
            "expected": expected,
            "actual": actual,
        }
        super().__init__(message, details=details)
        self.unit = unit
        self.target = target
        self.field = field


# 探查任务结局业务码（写入账本 task_trace.outcome_code）。
OUTCOME_SUCCESS = "T_SUCCESS"
OUTCOME_BUDGET_TIMEOUT = "T_BUDGET_TIMEOUT"
OUTCOME_DEPENDENCY_NOT_READY = "T_DEP_NOT_READY"
OUTCOME_CONFIG_MISSING = "T_CONFIG_MISSING"

ALL_OUTCOME_CODES = frozenset(
    {
        OUTCOME_SUCCESS,
        OUTCOME_BUDGET_TIMEOUT,
        OUTCOME_DEPENDENCY_NOT_READY,
        OUTCOME_CONFIG_MISSING,
    }
)
