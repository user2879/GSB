"""Business error codes and canonical failure records.

All recoverable, per-object failures are collected into an audit-able failure
list rather than raised; only fatal, whole-pipeline failures abort a run.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


E_TENSOR_LAYOUT = "E_TENSOR_LAYOUT"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_COMMAND_REF = "E_COMMAND_REF"
E_SELECT_UNCONVERGED = "E_SELECT_UNCONVERGED"
E_QBLOCK_BUDGET_EXHAUSTED = "E_QBLOCK_BUDGET_EXHAUSTED"
E_SCORE_BUDGET_EXHAUSTED = "E_SCORE_BUDGET_EXHAUSTED"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_NONFINITE_SCALE = "E_NONFINITE_SCALE"
E_ACC_OVERFLOW = "E_ACC_OVERFLOW"
E_PACK_TRUNCATED = "E_PACK_TRUNCATED"
E_INJECTION = "E_INJECTION"

ALL_CODES = (
    E_TENSOR_LAYOUT,
    E_CONFIG_INVALID,
    E_COMMAND_REF,
    E_SELECT_UNCONVERGED,
    E_QBLOCK_BUDGET_EXHAUSTED,
    E_SCORE_BUDGET_EXHAUSTED,
    E_SNAPSHOT_STALE,
    E_NONFINITE_SCALE,
    E_ACC_OVERFLOW,
    E_PACK_TRUNCATED,
    E_INJECTION,
)


@dataclass
class Failure:
    """One failure record; ``target`` identifies the affected object."""

    code: str
    stage: str
    target: str
    exception_type: str
    injection_id: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FailureList:
    failures: list[Failure] = field(default_factory=list)

    def add(
        self,
        code: str,
        stage: str,
        target: str,
        exception: BaseException | None = None,
        injection_id: str | None = None,
        detail: str = "",
    ) -> Failure:
        record = Failure(
            code=code,
            stage=stage,
            target=target,
            exception_type=type(exception).__name__ if exception else "",
            injection_id=injection_id,
            detail=detail,
        )
        self.failures.append(record)
        return record

    def codes(self) -> list[str]:
        return [f.code for f in self.failures]

    def to_list(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.failures]
