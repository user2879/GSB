"""引擎错误码定义。

所有错误都携带稳定的 ``code`` 与首个违规位置 ``location``，
便于结构化日志与验收脚本按错误码断言。
"""


class EngineError(Exception):
    def __init__(self, code, message, location=None, detail=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location
        self.detail = detail or {}

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "detail": self.detail,
        }


class SpecError(EngineError):
    def __init__(self, message, location, detail=None):
        super().__init__("E_SPEC", message, location, detail)


class ReplayError(EngineError):
    def __init__(self, message, location, detail=None):
        super().__init__("E_REPLAY", message, location, detail)


class BorrowCycleError(EngineError):
    def __init__(self, message, location, detail=None):
        super().__init__("E_BORROW", message, location, detail)


class DeadlineError(EngineError):
    def __init__(self, message, location, detail=None):
        super().__init__("E_DEADLINE", message, location, detail)


class DigestError(EngineError):
    def __init__(self, message, location, detail=None):
        super().__init__("E_DIGEST", message, location, detail)
