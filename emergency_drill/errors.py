"""业务错误码与异常定义。

所有可预期的业务失败均抛出 :class:`DrillError` 的子类，携带稳定的业务码、
首个违规位置与结构化上下文，便于 CLI、结构化日志与界面统一处理。
"""


class DrillError(Exception):
    code = "E_UNKNOWN"

    def __init__(self, message, location=None, **context):
        super().__init__(message)
        self.message = message
        self.location = location
        self.context = context

    def as_dict(self):
        data = {"code": self.code, "message": self.message, "location": self.location}
        data.update(self.context)
        return data

    def __str__(self):
        loc = f" @ {self.location}" if self.location else ""
        return f"[{self.code}] {self.message}{loc}"


class PlanValidationError(DrillError):
    code = "E_PLAN"


class CycleError(DrillError):
    code = "E_CYCLE"


class DependencyError(DrillError):
    code = "E_DEPEND"


class IncrementalMismatchError(DrillError):
    code = "E_INCR"


class SnapshotError(DrillError):
    code = "E_SNAPSHOT"


class ReplayError(DrillError):
    code = "E_REPLAY"


class LedgerError(DrillError):
    code = "E_LEDGER"


class StateError(DrillError):
    code = "E_STATE"
