"""统一错误码与带字段路径的校验异常。"""


class DiagError(Exception):
    """所有业务异常的基类，携带错误码与字段路径。"""

    code = "E_UNKNOWN"

    def __init__(self, message, path=None, value=None, details=None):
        super().__init__(message)
        self.message = message
        self.path = path
        self.value = value
        self.details = details or []

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "value": self.value,
            "details": list(self.details),
        }


class SpecError(DiagError):
    """清单强校验失败：整体拒绝加载。"""

    code = "E_SPEC"


class ReplayError(DiagError):
    """重放不可复现。"""

    code = "E_REPLAY"
