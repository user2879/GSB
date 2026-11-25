"""统一错误码与异常类型。

所有业务错误均携带稳定的机器可读码，便于账本复核与测试断言：
E_SPEC    输入声明/批次/参数非法
E_DIVERGE 双路径结论超出偏差界
E_LEASE   租约版本乐观校验失败或租约状态非法
E_REPLAY  独立重放与账本不一致
E_LEDGER  账本格式版本/摘要不一致或事务状态异常
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class SloError(Exception):
    code = "E_GENERIC"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if self.context:
            details = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
            return f"{self.code}: {self.message} ({details})"
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


class SpecError(SloError):
    code = "E_SPEC"


class DivergenceError(SloError):
    code = "E_DIVERGE"


class LeaseError(SloError):
    code = "E_LEASE"


class ReplayError(SloError):
    code = "E_REPLAY"


class LedgerError(SloError):
    code = "E_LEDGER"


@dataclass
class ErrorLocation:
    """错误定位：服务标识 + 文件行号（1 基）+ 批次序号（若适用）。"""

    service_id: str | None = None
    line: int | None = None
    minute: int | None = None
    field_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_context(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        if self.service_id is not None:
            ctx["service_id"] = self.service_id
        if self.line is not None:
            ctx["line"] = self.line
        if self.minute is not None:
            ctx["minute"] = self.minute
        if self.field_name is not None:
            ctx["field"] = self.field_name
        ctx.update(self.extra)
        return ctx
