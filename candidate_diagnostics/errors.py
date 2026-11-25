from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


E_SELECT_CODE_UNKNOWN = "E_SELECT_CODE_UNKNOWN"
E_FIELD_TRUNCATED = "E_FIELD_TRUNCATED"
E_CAND_COUNT_OVER = "E_CAND_COUNT_OVER"
E_CANDIDATE_ID_DUPLICATE = "E_CANDIDATE_ID_DUPLICATE"
E_REQUEST_ID_DUPLICATE = "E_REQUEST_ID_DUPLICATE"
E_SCORE_NON_FINITE = "E_SCORE_NON_FINITE"
E_AUX_NON_INTEGER = "E_AUX_NON_INTEGER"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_CHANNEL_UNKNOWN = "E_CHANNEL_UNKNOWN"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_REQUEST_QUOTA_EXCEEDED = "E_REQUEST_QUOTA_EXCEEDED"
E_WIDTH_OVERFLOW = "E_WIDTH_OVERFLOW"


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    source: str = ""
    request_id: str | None = None
    candidate_id: int | None = None
    line_no: int | None = None
    injection_id: str | None = None
    business_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiagnosticError(Exception):
    def __init__(self, failure: Failure):
        super().__init__(failure.message)
        self.failure = failure
