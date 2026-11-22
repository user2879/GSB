"""业务错误码与失败清单条目。"""


class EGWError(Exception):
    def __init__(self, code, message, context=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}


E_ROW_INVALID = "E_ROW_INVALID"
E_DIM_MISMATCH = "E_DIM_MISMATCH"
E_NONFINITE_VECTOR = "E_NONFINITE_VECTOR"
E_AGG_MODE_UNKNOWN = "E_AGG_MODE_UNKNOWN"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_ID_MISSING = "E_ID_MISSING"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"


ALL_CODES = (
    E_ROW_INVALID,
    E_DIM_MISMATCH,
    E_NONFINITE_VECTOR,
    E_AGG_MODE_UNKNOWN,
    E_CONFIG_INVALID,
    E_ID_MISSING,
    E_POINT_BUDGET_PENDING,
)


def failure(code, message, **context):
    entry = {"code": code, "message": message}
    entry.update(context)
    return entry


def sort_key(entry):
    """失败清单稳定排序键：业务码 -> 请求ID -> 位置 -> 来源 -> 消息。"""
    return (
        entry.get("code", ""),
        str(entry.get("request_id", "")),
        entry.get("position", -1) if entry.get("position") is not None else -1,
        str(entry.get("source", "")),
        str(entry.get("message", "")),
    )
