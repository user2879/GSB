"""Business error codes and per-entity failure records."""

from __future__ import annotations

from dataclasses import dataclass


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_PRECISION_INVALID = "E_PRECISION_INVALID"
E_OFFSET_OUT_OF_RANGE = "E_OFFSET_OUT_OF_RANGE"
E_NONFINITE_INPUT = "E_NONFINITE_INPUT"
E_LINE_OVERFLOW = "E_LINE_OVERFLOW"


class PipelineError(Exception):
    """Base class for pipeline errors carrying a stable business code."""

    code = "E_PIPELINE_ERROR"

    def __init__(self, message: str, *, entity_id: str | None = None,
                 stage: str | None = None):
        super().__init__(message)
        self.entity_id = entity_id
        self.stage = stage


class ConfigInvalidError(PipelineError):
    code = E_CONFIG_INVALID


class PrecisionInvalidError(PipelineError):
    code = E_PRECISION_INVALID


class OffsetOutOfRangeError(PipelineError):
    code = E_OFFSET_OUT_OF_RANGE


class NonFiniteInputError(PipelineError):
    code = E_NONFINITE_INPUT


class LineOverflowError(PipelineError):
    code = E_LINE_OVERFLOW


@dataclass(frozen=True)
class StageFailure:
    """One isolated per-entity/per-stage failure record."""

    entity_id: str
    stage: str
    code: str
    message: str
    sort_key: tuple

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "sort_key": [str(part) for part in self.sort_key],
        }
