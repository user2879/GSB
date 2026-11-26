from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Mapping


class Code(IntEnum):
    """Stable business codes.  Codes are part of the persisted contract."""

    OK = 0
    INVALID_SCHEMA = 1001
    DUPLICATE_DOMAIN = 1002
    DUPLICATE_DOCUMENT = 1003
    INVALID_WEIGHT = 1004
    WEIGHT_SUM_ZERO = 1005
    THRESHOLD_OUT_OF_RANGE = 1006
    INVALID_PRECISION = 1007
    INVALID_QUANTIZATION_TIER = 1008
    INVALID_SHARD_SIZE = 1009
    UNKNOWN_DOMAIN = 1010
    INVALID_RECEIPT = 1011
    INVALID_EDIT = 1012
    INVALID_DOCUMENT = 1013
    INVALID_BUDGET = 1014
    INVALID_TOLERANCE = 1015
    BUDGET_EXHAUSTED = 2001
    DOMAIN_QUOTA_EXHAUSTED = 2002
    DEDUP_COMPARE_LIMIT = 2003
    RECEIPT_OVERRIDE_REJECTED = 2004
    ZERO_WEIGHT_DOMAIN = 2005
    STATE_CORRUPT = 3001
    RECOVERY_MISMATCH = 3002


@dataclass(slots=True)
class PipelineError(Exception):
    code: Code
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{int(self.code)}:{self.code.name}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": int(self.code),
            "code_name": self.code.name,
            "message": self.message,
            "details": dict(self.details),
        }


def fail(code: Code, message: str, **details: Any) -> None:
    raise PipelineError(code=code, message=message, details=details)


def failure_record(
    code: Code,
    message: str,
    *,
    domain: str | None = None,
    shard_id: str | None = None,
    doc_id: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "code": int(code),
        "code_name": code.name,
        "message": message,
    }
    if domain is not None:
        record["domain"] = domain
    if shard_id is not None:
        record["shard_id"] = shard_id
    if doc_id is not None:
        record["doc_id"] = doc_id
    if details:
        record["details"] = details
    return record
