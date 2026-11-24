"""错误码与统一异常定义。

所有面向用户的拒绝都带稳定错误码，便于测试与文档逐项核对：

* ``E_SPEC``   声明文件 / 事件流强校验失败
* ``E_LEASE``  租约乐观版本不符或租约不存在 / 已终结
* ``E_ALIAS``  同一槽位被两个活跃租约同时持有
* ``E_DIVERGE`` 两条仲裁路径重放结论出现首个不一致
* ``E_LEDGER`` 账本格式版本或内容摘要不匹配
* ``E_PROTO``  回环 JSON 行协议层错误
* ``E_STATE``  状态机许可表不允许的迁移
"""

from __future__ import annotations


class FIError(Exception):
    """平台统一基类，携带稳定错误码与定位信息。"""

    code = "E_STATE"

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if self.context:
            details = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
            return f"[{self.code}] {message} ({details})" if False else f"[{self.code}] {self.message} ({details})"
        return f"[{self.code}] {self.message}"


class SpecError(FIError):
    code = "E_SPEC"

    def __init__(self, message: str, field: str, line: int | None = None, **context: object) -> None:
        super().__init__(message, field=field, line=line, **context)
        self.field = field
        self.line = line


class LeaseError(FIError):
    code = "E_LEASE"


class AliasError(FIError):
    code = "E_ALIAS"

    def __init__(self, message: str, lease_a: str, lease_b: str, slot: int, **context: object) -> None:
        super().__init__(message, lease_a=lease_a, lease_b=lease_b, slot=slot, **context)
        self.lease_a = lease_a
        self.lease_b = lease_b
        self.slot = slot


class DivergenceError(FIError):
    code = "E_DIVERGE"

    def __init__(self, message: str, seq: int, instance: str, field: str, **context: object) -> None:
        super().__init__(message, seq=seq, instance=instance, field=field, **context)
        self.seq = seq
        self.instance = instance
        self.field = field


class LedgerError(FIError):
    code = "E_LEDGER"


class ProtocolError(FIError):
    code = "E_PROTO"


class StateError(FIError):
    code = "E_STATE"
