"""业务异常与统一业务码。

所有可预期的领域错误都抛出 :class:`BenchError`，携带稳定的业务码，
便于工作台界面与演示脚本统一展示。
"""
from __future__ import annotations

from dataclasses import dataclass


class BenchError(Exception):
    """工作台领域错误基类。"""

    code: str = "E_UNKNOWN"

    def __init__(self, message: str, details: dict | None = None,
                 *, extra: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or extra or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message,
                "details": self.details}


class SchemaError(BenchError):
    """段定义/块表强校验失败。"""

    code = "E_SCHEMA"


class PayloadError(BenchError):
    """块载荷截断、摘要不匹配等持久化数据损坏。"""

    code = "E_PAYLOAD"


class EncodingError(BenchError):
    """块头编码声明与实际载荷不匹配。"""

    code = "E_ENCODING"


class BudgetError(BenchError):
    """三重预算约束被违反或预算为零导致无法调度。"""

    code = "E_BUDGET"


class DivergenceError(BenchError):
    """账本记录与权威重算结果不一致。"""

    code = "E_DIVERGE"


class SnapshotError(BenchError):
    """快照版本或内容摘要不匹配。"""

    code = "E_SNAPSHOT"


@dataclass(frozen=True)
class Issue:
    """单条强校验问题，携带出错列与块序号。"""

    code: str
    message: str
    column: str | None = None
    block_index: int | None = None
    row_group: int | None = None

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message,
                "column": self.column, "block_index": self.block_index,
                "row_group": self.row_group}
