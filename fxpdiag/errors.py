"""业务错误码与失败清单条目。

所有非法输入都按业务码分流到失败清单（failure list），管线不会静默跳过
任何条目；致命结构错误抛出 :class:`PipelineError`，行级错误以
:class:`FailureRecord` 形式收集。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """业务错误码枚举（字符串形式便于写入 JSON 报告）。"""

    E_DUPLICATE_LAYER = "E_DUPLICATE_LAYER"
    E_SHAPE_INVALID = "E_SHAPE_INVALID"
    E_NON_FINITE = "E_NON_FINITE"
    E_FIELD_RANGE = "E_FIELD_RANGE"
    E_BITWIDTH_ENUM = "E_BITWIDTH_ENUM"
    E_LAYER_CYCLE = "E_LAYER_CYCLE"
    E_UNKNOWN_REF = "E_UNKNOWN_REF"
    E_V1_LINE_MALFORMED = "E_V1_LINE_MALFORMED"
    E_V2_SCHEMA = "E_V2_SCHEMA"
    E_COMMAND_UNKNOWN = "E_COMMAND_UNKNOWN"
    E_BUDGET_PENDING = "E_OP_BUDGET_PENDING"
    E_INJECTED = "E_INJECTED"


class PipelineError(Exception):
    """致命错误（无法继续构建管线），携带业务码。"""

    def __init__(self, code: ErrorCode, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": self.message, "context": self.context}


@dataclass
class FailureRecord:
    """失败清单中的单条记录，顺序即发现顺序。"""

    code: ErrorCode
    source: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "source": self.source,
            "message": self.message,
            "context": self.context,
        }


class FailureList:
    """保持插入顺序的失败清单，供报告与一致性比对使用。"""

    def __init__(self) -> None:
        self._records: list[FailureRecord] = []

    def add(
        self,
        code: ErrorCode,
        source: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self._records.append(FailureRecord(code, source, message, context or {}))

    def extend(self, records: list[FailureRecord]) -> None:
        self._records.extend(records)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records]

    def codes(self) -> list[str]:
        return [record.code.value for record in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)
