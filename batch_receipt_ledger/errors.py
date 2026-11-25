"""统一错误类型与错误码。

所有可预期业务错误都继承 :class:`BatchError`，携带稳定错误码、
首个非法位置（JSON Pointer 风格路径）与字段名，便于唯一复核。
"""


class BatchError(Exception):
    """带错误码、位置与字段名的基类异常。"""

    code = "E_UNKNOWN"

    def __init__(self, message, location=None, field=None):
        super().__init__(message)
        self.message = message
        self.location = location
        self.field = field

    def __str__(self):
        parts = [self.code, self.message]
        if self.location is not None:
            parts.append(f"at {self.location}")
        if self.field is not None:
            parts.append(f"field={self.field}")
        return " ".join(parts)

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "field": self.field,
        }


class SpecError(BatchError):
    """声明/请求流/配额事件校验失败。"""

    code = "E_SPEC"


class FieldError(BatchError):
    """回执字段语义校验失败（耗时非负、名次范围等）。"""

    code = "E_FIELD"


class RefError(BatchError):
    """引用校验失败（重复投递引用指向不存在的请求等）。"""

    code = "E_REF"


class PortError(BatchError):
    """回环端口绑定/占用失败。"""

    code = "E_PORT"


class TimeoutError_(BatchError):
    """真实套接字握手或读取超时。"""

    code = "E_TIMEOUT"


class LeaseError(BatchError):
    """租约版本不符或租约非法。"""

    code = "E_LEASE"


class LedgerError(BatchError):
    """账本重放校验失败或快照损坏。"""

    code = "E_LEDGER"


class WorkerError(BatchError):
    """工作进程意外退出等运行时错误。"""

    code = "E_WORKER"
