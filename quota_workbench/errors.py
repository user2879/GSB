"""统一错误码：E_FIELD / E_ORDER / E_SNAPSHOT / E_REPLAY。"""


class WorkbenchError(Exception):
    """带错误码与结构化细节的工作台异常。"""

    def __init__(self, code, message, details=None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}
