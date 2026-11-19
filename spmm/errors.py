"""业务错误码与异常定义。"""

E_CSR_INVALID = "E_CSR_INVALID"
E_NONFINITE = "E_NONFINITE"
E_SHAPE_MISMATCH = "E_SHAPE_MISMATCH"
E_BLOCK_INVALID = "E_BLOCK_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_BUDGET_ZERO = "E_BUDGET_ZERO"
E_RECOMPUTE_PENDING = "E_RECOMPUTE_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"


class SpmmError(Exception):
    """携带业务码与违规明细的组件异常。"""

    def __init__(self, code, message, violations=None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.violations = violations or []

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "violations": self.violations,
        }
