"""Stable business error codes used across the diagnostic component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# Every public error code lives here so reports can map injections to codes.
E_INPUT_INVALID = "E_INPUT_INVALID"
E_STAGE_BUDGET_EXCEEDED = "E_STAGE_BUDGET_EXCEEDED"
E_ARTIFACT_BUDGET_PENDING = "E_ARTIFACT_BUDGET_PENDING"
E_SNAPSHOT_VERSION_MISMATCH = "E_SNAPSHOT_VERSION_MISMATCH"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_INJECTION_HIT = "E_INJECTION_HIT"


@dataclass
class Violation:
    """A single validation finding with a stable locator."""

    code: str
    locator: str
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "locator": self.locator, "message": self.message}


class FcdError(Exception):
    """Base class for all component errors."""

    code = "E_FCD_ERROR"


class InputInvalidError(FcdError):
    """Raised after collecting *all* validation violations."""

    code = E_INPUT_INVALID

    def __init__(self, violations: List[Violation]):
        self.violations = violations
        detail = "; ".join(f"{v.locator}: {v.message}" for v in violations)
        super().__init__(f"{E_INPUT_INVALID}: {detail}")


class StageBudgetExceeded(FcdError):
    code = E_STAGE_BUDGET_EXCEEDED

    def __init__(self, stage: str, used: int, quota: int):
        self.stage = stage
        self.used = used
        self.quota = quota
        super().__init__(
            f"{E_STAGE_BUDGET_EXCEEDED}: stage={stage} used={used} quota={quota}"
        )


class SnapshotVersionMismatch(FcdError):
    code = E_SNAPSHOT_VERSION_MISMATCH


class SnapshotCorrupt(FcdError):
    code = E_SNAPSHOT_CORRUPT

