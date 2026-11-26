"""错误码与异常体系。

所有业务错误都携带稳定错误码，便于上层脚本与测试精确断言。
"""
from __future__ import annotations

from typing import Any


class DispatchError(Exception):
    """所有业务异常的基类。"""

    code = "E_INTERNAL"

    def __init__(self, message: str, *, location: str | None = None,
                 field: str | None = None, **extra: Any) -> None:
        super().__init__(message)
        self.message = message
        self.location = location
        self.field = field
        self.extra = extra

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.location is not None:
            data["location"] = self.location
        if self.field is not None:
            data["field"] = self.field
        data.update(self.extra)
        return data


class SpecError(DispatchError):
    """规格文件校验失败：首个违规位置与字段名。"""

    code = "E_SPEC"


class PortError(DispatchError):
    """回环端口绑定/占用失败。"""

    code = "E_PORT"


class TimeoutError(DispatchError):
    """真实 TCP 套接字握手或读取超时。"""

    code = "E_TIMEOUT"


class FieldError(DispatchError):
    """回执字段缺失或类型不符。"""

    code = "E_FIELD"


class LeaseError(DispatchError):
    """令牌乐观版本校验失败。"""

    code = "E_LEASE"


class DiffError(DispatchError):
    """增量结果与全量重建结果不一致。"""

    code = "E_DIFF"


class ReplayError(DispatchError):
    """强制终止恢复后重放出现首个不一致。"""

    code = "E_REPLAY"
