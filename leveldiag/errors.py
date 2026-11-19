"""业务错误码。

E_SPEC_INVALID       规格字段缺失/越界/权重归一化偏差超容忍等
E_QUANT_BOUNDS       量化值越出声明有效域，或量化误差超出声明上界
E_LEDGER_STALE       账本引用的规格/网格/映射表哈希与当前工件不一致
E_LEDGER_MISMATCH    独立复算器复算结论与生成器账本不一致
E_BOUNDARY_AMBIGUOUS 严格模式下存在落在误差区间内的不确定判定
E_CLOSURE_MISMATCH   增量重生成实际重算集合与声明闭包不一致
"""

from __future__ import annotations


class ErrorCode:
    E_SPEC_INVALID = "E_SPEC_INVALID"
    E_QUANT_BOUNDS = "E_QUANT_BOUNDS"
    E_LEDGER_STALE = "E_LEDGER_STALE"
    E_LEDGER_MISMATCH = "E_LEDGER_MISMATCH"
    E_BOUNDARY_AMBIGUOUS = "E_BOUNDARY_AMBIGUOUS"
    E_CLOSURE_MISMATCH = "E_CLOSURE_MISMATCH"


class LevelDiagError(Exception):
    """带业务错误码与结构化明细的异常。"""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}
