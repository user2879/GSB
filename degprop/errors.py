"""业务错误码与异常定义。

所有可预期的领域错误均以 :class:`DegPropError` 抛出，携带稳定业务码、
首个违规字段、行号与依据说明，便于上层做整体拒绝与结构化日志记录。
"""
from __future__ import annotations

from dataclasses import dataclass


class DegPropError(Exception):
    """组件领域错误基类。"""

    code = "E_INTERNAL"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        line: int | None = None,
        detail: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.line = line
        self.detail = detail or {}

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "line": self.line,
            "detail": self.detail,
        }


class TopologyError(DegPropError):
    """拓扑/事件流强校验失败（整体拒绝，不产生部分加载结果）。"""

    code = "E_TOPO"


class CycleError(DegPropError):
    """注入或检测到服务依赖图成环。"""

    code = "E_CYCLE"


class CapacityOverflowError(DegPropError):
    """降级容量上限与故障域配额乘积越过可表示范围。"""

    code = "E_CAPACITY"


class SnapshotError(DegPropError):
    """快照格式版本或内容摘要不符，触发整体替换。"""

    code = "E_SNAPSHOT"


class ReplayError(DegPropError):
    """恢复重放后首个不一致定位失败。"""

    code = "E_REPLAY"


@dataclass(frozen=True)
class BusinessCode:
    TOPO = "E_TOPO"
    CYCLE = "E_CYCLE"
    CAPACITY = "E_CAPACITY"
    SNAPSHOT = "E_SNAPSHOT"
    REPLAY = "E_REPLAY"
    OK = "E_OK"
