"""业务错误码与结构化异常。"""

from __future__ import annotations


class Code:
    # 加载校验层
    E_TENSOR_MISSING = "E_TENSOR_MISSING"
    E_SHAPE_DATA_MISMATCH = "E_SHAPE_DATA_MISMATCH"
    E_PAD_RULE_VIOLATION = "E_PAD_RULE_VIOLATION"
    E_NONFINITE = "E_NONFINITE"
    E_HEAD_DIM_MISMATCH = "E_HEAD_DIM_MISMATCH"
    E_FRAC_BITS_OUT_OF_RANGE = "E_FRAC_BITS_OUT_OF_RANGE"
    E_QUANT_LEVEL_INVALID = "E_QUANT_LEVEL_INVALID"
    E_WORKERS_OUT_OF_RANGE = "E_WORKERS_OUT_OF_RANGE"
    # 整数内核（记录型，非致命）
    E_ACCUMULATOR_OVERFLOW = "E_ACCUMULATOR_OVERFLOW"
    # 调度预算
    E_NODE_BUDGET_EXHAUSTED = "E_NODE_BUDGET_EXHAUSTED"
    E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
    # 异常注入
    E_INJECTED_FAULT = "E_INJECTED_FAULT"
    # 快照恢复
    E_SNAPSHOT_VERSION_MISMATCH = "E_SNAPSHOT_VERSION_MISMATCH"


class OperatorError(Exception):
    """携带业务码与块定位信息的异常。"""

    def __init__(self, code, message, stage=None, query_block=None,
                 key_block=None, injection_id=None):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.stage = stage
        self.query_block = query_block
        self.key_block = key_block
        self.injection_id = injection_id

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "query_block": self.query_block,
            "key_block": self.key_block,
            "injection_id": self.injection_id,
        }
