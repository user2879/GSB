"""Business error codes and a typed pipeline exception / failure record."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Business error code enumeration (stable strings are part of the report ABI).
# ---------------------------------------------------------------------------
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_IMAGE_FORMAT = "E_IMAGE_FORMAT"
E_ASSERTION_SCHEMA = "E_ASSERTION_SCHEMA"
E_RELATION_REF = "E_RELATION_REF"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_INJECTION_HIT = "E_INJECTION_HIT"

ALL_ERROR_CODES = (
    E_CONFIG_INVALID,
    E_IMAGE_FORMAT,
    E_ASSERTION_SCHEMA,
    E_RELATION_REF,
    E_SNAPSHOT_STALE,
    E_INJECTION_HIT,
)


class PipelineError(Exception):
    """Raised when a stage encounters a classified, recoverable business fault.

    ``stage`` is one of the four pipeline stage names; ``region`` may be None
    for sample-level or configuration-level faults.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        sample: Optional[str] = None,
        region: Optional[int] = None,
        injection_id: Optional[str] = None,
    ) -> None:
        if code not in ALL_ERROR_CODES:
            raise ValueError(f"unknown business error code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.sample = sample
        self.region = region
        self.injection_id = injection_id

    def to_record(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "sample": self.sample,
            "region": self.region,
            "injection_id": self.injection_id,
        }


@dataclass(frozen=True)
class FailureRecord:
    """One classified failure. Ordered by a stable key inside the failure list."""

    sample: Optional[str]
    region: Optional[int]
    stage: str
    code: str
    message: str
    injection_id: Optional[str] = None

    @property
    def sort_key(self):
        # sample name, region (-1 for sample-level), stage order, code.
        return (
            self.sample or "",
            -1 if self.region is None else self.region,
            self.stage,
            self.code,
            self.message,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def failure_from_error(err: PipelineError) -> FailureRecord:
    return FailureRecord(
        sample=err.sample,
        region=err.region,
        stage=err.stage,
        code=err.code,
        message=err.message,
        injection_id=err.injection_id,
    )
