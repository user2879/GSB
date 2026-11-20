"""业务错误码与异常类型。

所有可预期的业务失败都抛出 ArbiterError 子类，携带稳定业务码、
首个出错对象标识与字段，便于结构化日志追溯。
"""


class ArbiterError(Exception):
    code = "E_UNKNOWN"

    def __init__(self, message, *, ref_id=None, field=None, detail=None):
        super().__init__(message)
        self.message = message
        self.ref_id = ref_id
        self.field = field
        self.detail = detail or {}

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "ref_id": self.ref_id,
            "field": self.field,
            "detail": self.detail,
        }


class SpecError(ArbiterError):
    """声明 / 请求流强校验失败，整体拒绝（E_SPEC）。"""

    code = "E_SPEC"


class OrderError(ArbiterError):
    """注入：请求流到达时刻倒退（E_ORDER）。"""

    code = "E_ORDER"


class PriorityError(ArbiterError):
    """注入：资源池优先级序引用缺失租户（E_PRIORITY）。"""

    code = "E_PRIORITY"


class LedgerError(ArbiterError):
    """注入 / 损坏：账本内容摘要不符（E_LEDGER）。"""

    code = "E_LEDGER"


class ReplayError(ArbiterError):
    """快照重放不可复现（E_REPLAY），携带首个不一致请求与字段。"""

    code = "E_REPLAY"
