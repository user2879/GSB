"""业务错误码定义。"""


class BizError(Exception):
    """携带业务码的异常，加载校验与运行期判定统一使用。"""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


E_TASK_INVALID = "E_TASK_INVALID"
E_FEATURE_MISSING = "E_FEATURE_MISSING"
E_ENCODER_SHAPE = "E_ENCODER_SHAPE"
E_CONTRIB_INVALID = "E_CONTRIB_INVALID"
E_PLAN_INVALID = "E_PLAN_INVALID"
E_RETRAIN_BUDGET_PENDING = "E_RETRAIN_BUDGET_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_CLOSURE_VIOLATION = "E_CLOSURE_VIOLATION"
