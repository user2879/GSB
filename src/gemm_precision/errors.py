from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


E_NONFINITE_INPUT = "E_NONFINITE_INPUT"
E_SHAPE_MISMATCH = "E_SHAPE_MISMATCH"
E_BLOCK_INVALID = "E_BLOCK_INVALID"
E_TOLERANCE_INVALID = "E_TOLERANCE_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_DIAGNOSTIC_MISMATCH = "E_DIAGNOSTIC_MISMATCH"
E_CONCURRENCY_MISMATCH = "E_CONCURRENCY_MISMATCH"
E_INTERNAL = "E_INTERNAL"

BUSINESS_CODE_CATALOG: dict[str, str] = {
    E_NONFINITE_INPUT: "A/B/C/alpha/beta contains NaN or infinity.",
    E_SHAPE_MISMATCH: "Matrix, batch, or output shape is not compatible.",
    E_BLOCK_INVALID: "A block dimension is zero or negative.",
    E_TOLERANCE_INVALID: "Tolerance is negative, non-finite, or both tolerances are zero.",
    E_CONFIG_INVALID: "Experiment configuration is inconsistent or incomplete.",
    E_DIAGNOSTIC_MISMATCH: "Independent trace replay disagrees with the kernel output.",
    E_CONCURRENCY_MISMATCH: "Thread-pool output differs from the serial reference execution.",
    E_INTERNAL: "Unexpected internal failure captured for failure isolation.",
}


@dataclass
class BusinessError(RuntimeError):
    code: str
    stage: str
    message: str
    shape: Any = None
    batch_index: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    injection_id: str | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "business_code": self.code,
            "stage": self.stage,
            "message": self.message,
        }
        if self.shape is not None:
            record["shape"] = self.shape
        if self.batch_index is not None:
            record["batch_index"] = self.batch_index
        if self.injection_id is not None:
            record["injection_id"] = self.injection_id
        if self.details:
            record["details"] = dict(self.details)
        return record
