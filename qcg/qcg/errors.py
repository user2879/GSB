"""Business error codes and fault records for the quantized conv graph app."""
from dataclasses import dataclass, asdict
from typing import Optional

E_GRAPH_INVALID = "E_GRAPH_INVALID"
E_LOSSY_DOWNGRADE = "E_LOSSY_DOWNGRADE"
E_ACC_OVERFLOW = "E_ACC_OVERFLOW"
E_NODE_BUDGET_EXHAUSTED = "E_NODE_BUDGET_EXHAUSTED"
E_GLOBAL_BUDGET_PENDING = "E_GLOBAL_BUDGET_PENDING"
E_RECOMPUTE_PENDING = "E_RECOMPUTE_PENDING"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_QUANT_PACK_INVALID = "E_QUANT_PACK_INVALID"
E_INJECTION_FAULT = "E_INJECTION_FAULT"


class QCGError(Exception):
    def __init__(self, code: str, message: str, node_id: Optional[str] = None,
                 field: Optional[str] = None, stage: Optional[str] = None,
                 injection_id: Optional[str] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.node_id = node_id
        self.field = field
        self.stage = stage
        self.injection_id = injection_id

    def to_record(self, seq: Optional[int] = None) -> dict:
        rec = {"code": self.code, "message": self.message,
               "node_id": self.node_id, "field": self.field,
               "stage": self.stage, "injection_id": self.injection_id}
        if seq is not None:
            rec["seq"] = seq
        return rec


@dataclass
class FaultRecord:
    code: str
    message: str
    node_id: Optional[str] = None
    field: Optional[str] = None
    stage: Optional[str] = None
    injection_id: Optional[str] = None
    seq: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)
