from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Failure:
    identifier: str
    stage: str
    code: str
    exception_type: str
    message: str
    source: Optional[str] = None
    line: Optional[int] = None
    injection_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "stage": self.stage,
            "code": self.code,
            "exception_type": self.exception_type,
            "message": self.message,
            "source": self.source,
            "line": self.line,
            "injection_id": self.injection_id,
        }

    @property
    def stable_key(self) -> tuple:
        return (
            self.stage,
            self.identifier,
            self.code,
            self.exception_type,
            self.message,
            self.source or "",
            -1 if self.line is None else self.line,
            self.injection_id or "",
        )


class PipelineError(Exception):
    code = "E_PIPELINE"
    stage = "pipeline"

    def __init__(self, message: str, *, injection_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.injection_id = injection_id


class ConfigError(PipelineError):
    code = "E_CONFIG_INVALID"
    stage = "config"


class SnapshotMismatchError(PipelineError):
    code = "E_CONFIG_MISMATCH"
    stage = "snapshot"


class RecordInvalidError(PipelineError):
    code = "E_RECORD_INVALID"
    stage = "normalize"


class RecallLimitError(PipelineError):
    code = "E_RECALL_LIMIT_ZERO"
    stage = "retrieval_scoring"


class QueryBudgetExhaustedError(PipelineError):
    code = "E_QUERY_BUDGET_EXHAUSTED"
    stage = "retrieval_scoring"


class GlobalBudgetExhaustedError(PipelineError):
    code = "E_BUDGET_EXHAUSTED"
    stage = "retrieval_scoring"


class MissingTranslationInjectionError(PipelineError):
    code = "E_TRANSLATION_MISSING"
    stage = "retrieval_scoring"


class NonFiniteConfidenceInjectionError(PipelineError):
    code = "E_RECORD_INVALID"
    stage = "retrieval_scoring"


class RelevanceGradeError(PipelineError):
    code = "E_RELEVANCE_GRADE_OUT_OF_RANGE"
    stage = "evaluation"
