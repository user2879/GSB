"""Business error codes used across the extraction application."""

from __future__ import annotations

from dataclasses import dataclass


class ErrorCode:
    E_DOC_INVALID = "E_DOC_INVALID"
    E_JSON_PARSE = "E_JSON_PARSE"
    E_ENTITY_TYPE_UNKNOWN = "E_ENTITY_TYPE_UNKNOWN"
    E_POSITION_OUT_OF_RANGE = "E_POSITION_OUT_OF_RANGE"
    E_ARGT_MISSING = "E_ARGT_MISSING"
    E_INTERACT_BUDGET_PENDING = "E_INTERACT_BUDGET_PENDING"


@dataclass(frozen=True)
class Failure:
    """A single failure-list entry with stable, sortable fields."""

    doc_id: str
    stage: str
    code: str
    detail: str

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "stage": self.stage,
            "code": self.code,
            "detail": self.detail,
        }
