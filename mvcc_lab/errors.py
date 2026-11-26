"""约定业务码与统一异常类型。"""


class MvccError(Exception):
    """所有组件级错误的基类，携带稳定业务码。"""

    code = "E_UNKNOWN"

    def __init__(self, message: str, *, object_id=None, field=None, detail=None):
        super().__init__(message)
        self.message = message
        self.object_id = object_id
        self.field = field
        self.detail = detail

    def to_dict(self) -> dict:
        payload = {"code": self.code, "message": self.message}
        if self.object_id is not None:
            payload["object_id"] = self.object_id
        if self.field is not None:
            payload["field"] = self.field
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


class SpecError(MvccError):
    """声明文件非法（业务要求 1）。"""

    code = "E_SPEC"


class PortError(MvccError):
    """回环端口无法绑定/占用（业务要求 3）。"""

    code = "E_PORT"


class TimeoutError_(MvccError):
    """真实套接字握手或读取超时（业务要求 3）。"""

    code = "E_TIMEOUT"


class FieldError_(MvccError):
    """回执字段缺失（业务要求 3）。"""

    code = "E_FIELD"


class ReplayError(MvccError):
    """强制终止恢复后的重放结论不一致（业务要求 8）。"""

    code = "E_REPLAY"


class LedgerError(MvccError):
    """账本格式或状态错误。"""

    code = "E_LEDGER"
