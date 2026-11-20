from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SEGMENT_LEN_INVALID = "E_SEGMENT_LEN_INVALID"
E_SEGMENT_MISMATCH = "E_SEGMENT_MISMATCH"
E_RATIO_OUT_OF_RANGE = "E_RATIO_OUT_OF_RANGE"
E_BAND_INVALID = "E_BAND_INVALID"
E_THRESHOLD_INVALID = "E_THRESHOLD_INVALID"
E_CONCURRENCY_INVALID = "E_CONCURRENCY_INVALID"
E_QUANT_INVALID = "E_QUANT_INVALID"
E_NONFINITE_SAMPLE = "E_NONFINITE_SAMPLE"
E_FEATURE_DEPENDENCY = "E_FEATURE_DEPENDENCY"
E_ALGO_PARAM_INVALID = "E_ALGO_PARAM_INVALID"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"


class PipelineError(Exception):
    """A business error that is reported and does not abort the whole run."""

    def __init__(self, code: str, detail: str, *, injected_id: str | None = None):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.injected_id = injected_id


@dataclass(frozen=True)
class Failure:
    device_id: str | None
    segment_index: int | None
    algorithm: str | None
    stage: str
    code: str
    detail: str
    injected_id: str | None = None

    def sort_key(self) -> tuple[Any, ...]:
        stage_rank = {
            "load": 0,
            "feature_extraction": 1,
            "scoring": 2,
            "threshold": 3,
            "summary": 4,
        }
        return (
            self.device_id or "",
            -1 if self.segment_index is None else self.segment_index,
            self.algorithm or "",
            stage_rank.get(self.stage, 9),
            self.stage,
            self.code,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "segment_index": self.segment_index,
            "algorithm": self.algorithm,
            "stage": self.stage,
            "code": self.code,
            "detail": self.detail,
            "injected_id": self.injected_id,
        }


def failure_from_error(
    error: Exception,
    *,
    device_id: str | None,
    segment_index: int | None,
    algorithm: str | None,
    stage: str,
) -> Failure:
    if isinstance(error, PipelineError):
        return Failure(
            device_id,
            segment_index,
            algorithm,
            stage,
            error.code,
            error.detail,
            error.injected_id,
        )
    return Failure(
        device_id,
        segment_index,
        algorithm,
        stage,
        E_CONFIG_INVALID,
        f"unexpected {type(error).__name__}: {error}",
        None,
    )
