"""业务错误码与可定位异常。

所有业务异常都带稳定错误码、字段名与上下文信息，CLI/GUI 可直接展示，
测试也以错误码而非异常文案作为断言依据。
"""
from __future__ import annotations

from typing import Any, Mapping


E_FORMAT_UNSUPPORTED = "E_FORMAT_UNSUPPORTED"
E_SCHEMA_VERSION = "E_SCHEMA_VERSION"
E_FIELD_MISSING = "E_FIELD_MISSING"
E_CHANNEL_MISMATCH = "E_CHANNEL_MISMATCH"
E_CACHE_HASH = "E_CACHE_HASH"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_PARAM_RANGE = "E_PARAM_RANGE"

ALL_ERROR_CODES = (
    E_FORMAT_UNSUPPORTED,
    E_SCHEMA_VERSION,
    E_FIELD_MISSING,
    E_CHANNEL_MISMATCH,
    E_CACHE_HASH,
    E_SNAPSHOT_STALE,
    E_PARAM_RANGE,
)


class RepairError(Exception):
    """带业务错误码的异常。

    code: 稳定业务码（见模块级常量）
    field: 触发问题的字段名/阶段名，便于定位
    context: 任意可 JSON 化的补充信息
    """

    def __init__(
        self,
        code: str,
        message: str,
        field: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if code not in ALL_ERROR_CODES:
            raise ValueError(f"unknown error code: {code}")
        super().__init__(message)
        self.code = code
        self.field = field
        self.context = dict(context or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "field": self.field,
            "context": self.context,
        }

    def __str__(self) -> str:  # pragma: no cover - 仅展示
        loc = f" [field={self.field}]" if self.field else ""
        return f"{self.code}{loc}: {self.args[0]}"
