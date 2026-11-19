"""业务错误码定义。

所有错误都带稳定的 code、出错键与可定位的字段路径，便于结构化日志
与诊断界面展示，也便于测试做确定性断言。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    key: str | None = None
    field_path: str | None = None
    line: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.key is not None:
            payload["key"] = self.key
        if self.field_path is not None:
            payload["field_path"] = self.field_path
        if self.line is not None:
            payload["line"] = self.line
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:
        parts = [self.code, self.message]
        if self.line is not None:
            parts.append(f"line={self.line}")
        if self.field_path:
            parts.append(f"path={self.field_path}")
        if self.key is not None:
            parts.append(f"key={self.key}")
        return " ".join(parts)


class EGraph(BusinessError):
    def __init__(self, message: str, *, key: str | None = None,
                 field_path: str | None = None, line: int | None = None,
                 cycle: Sequence[str] | None = None,
                 details: dict[str, Any] | None = None) -> None:
        merged = dict(details or {})
        if cycle is not None:
            merged["cycle"] = list(cycle)
        super().__init__("E_GRAPH", message, key=key, field_path=field_path,
                         line=line, details=merged)


class EFormat(BusinessError):
    def __init__(self, message: str, *, key: str | None = None,
                 field_path: str | None = None, line: int | None = None,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__("E_FORMAT", message, key=key, field_path=field_path,
                         line=line, details=details or {})


class EClock(BusinessError):
    def __init__(self, message: str, *, key: str | None = None,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__("E_CLOCK", message, key=key, details=details or {})


class ESnapshot(BusinessError):
    def __init__(self, message: str, *, key: str | None = None,
                 field_path: str | None = None, line: int | None = None,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__("E_SNAPSHOT", message, key=key,
                         field_path=field_path, line=line,
                         details=details or {})


class EInherit(BusinessError):
    def __init__(self, message: str, *, key: str | None = None,
                 field_path: str | None = None, line: int | None = None,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__("E_INHERIT", message, key=key,
                         field_path=field_path, line=line,
                         details=details or {})


class EReplay(BusinessError):
    def __init__(self, message: str, *, key: str | None = None,
                 field_path: str | None = None,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__("E_REPLAY", message, key=key,
                         field_path=field_path, details=details or {})
