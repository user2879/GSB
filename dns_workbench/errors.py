"""统一错误码与结构化异常。

错误码约定：
E_WIRE     线格式报文非法（含出错偏移 offset）
E_FORMAT   记录/应答格式非法（未知字段、缺字段、TTL 越界等，附字段列表）
E_VER      版本错误（未知版本、来源序列号倒退、版本不匹配）
E_ZONE     区文件损坏
E_UPSTREAM 本地回环上游不可达
E_CHAIN    上游应答链不自洽
E_TXN      上游应答事务标识错误
E_SNAPSHOT 快照恢复校验不一致
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class WorkbenchError(Exception):
    """所有工作台异常的基类，携带稳定错误码与结构化上下文。"""

    code = "E_UNKNOWN"

    def __init__(
        self,
        message: str,
        *,
        fields: list[str] | None = None,
        offset: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.fields: list[str] = sorted(fields or [])
        self.offset = offset
        self.context: dict[str, Any] = dict(context or {})

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.fields:
            data["fields"] = list(self.fields)
        if self.offset is not None:
            data["offset"] = self.offset
        if self.context:
            data["context"] = dict(self.context)
        return data


class WireError(WorkbenchError):
    code = "E_WIRE"


class FormatError(WorkbenchError):
    code = "E_FORMAT"


class VersionError(WorkbenchError):
    code = "E_VER"


class ZoneError(WorkbenchError):
    code = "E_ZONE"


class UpstreamError(WorkbenchError):
    code = "E_UPSTREAM"


class ChainError(WorkbenchError):
    code = "E_CHAIN"


class TxnError(WorkbenchError):
    code = "E_TXN"


class SnapshotError(WorkbenchError):
    code = "E_SNAPSHOT"


@dataclass
class StructuredLogRecord:
    """结构化日志记录（同时可序列化为 JSON 行）。"""

    code: str
    message: str
    level: str = "ERROR"
    fields: list[str] = field(default_factory=list)
    offset: int | None = None
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, exc: WorkbenchError, level: str = "ERROR") -> "StructuredLogRecord":
        return cls(
            code=exc.code,
            message=exc.message,
            level=level,
            fields=list(exc.fields),
            offset=exc.offset,
            context=dict(exc.context),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"level": self.level, "code": self.code, "message": self.message}
        if self.fields:
            data["fields"] = list(self.fields)
        if self.offset is not None:
            data["offset"] = self.offset
        if self.context:
            data["context"] = dict(self.context)
        return data
