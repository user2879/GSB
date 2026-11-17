"""Business errors and stable failure records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_INPUT_NON_FINITE = "E_INPUT_NON_FINITE"
E_INPUT_INVALID = "E_INPUT_INVALID"
E_CATEGORY_UNMAPPED = "E_CATEGORY_UNMAPPED"
E_HYPERPARAM_OUT_OF_RANGE = "E_HYPERPARAM_OUT_OF_RANGE"
E_DBSCAN_RADIUS_INVALID = "E_DBSCAN_RADIUS_INVALID"
E_INJECTION_NOT_TRIGGERED = "E_INJECTION_NOT_TRIGGERED"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_SNAPSHOT_MISSING = "E_SNAPSHOT_MISSING"
E_MODEL_FAILED = "E_MODEL_FAILED"
E_EVALUATION_FAILED = "E_EVALUATION_FAILED"
E_NONDETERMINISTIC = "E_NONDETERMINISTIC"
E_INTERNAL = "E_INTERNAL"


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    stage: str
    injection_id: str | None = None
    candidate_key: str | None = None
    path: str | None = None
    fatal: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
            "injection_id": self.injection_id,
            "candidate_key": self.candidate_key,
            "path": self.path,
            "fatal": self.fatal,
        }
        if self.extra:
            result["extra"] = self.extra
        return result

    def sort_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.code,
            self.stage,
            self.injection_id or "",
            self.candidate_key or "",
            self.message,
        )


class BusinessError(Exception):
    """A failure that must be represented as an E_* business code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        injection_id: str | None = None,
        candidate_key: str | None = None,
        path: str | None = None,
        fatal: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure = Failure(
            code=code,
            message=message,
            stage=stage,
            injection_id=injection_id,
            candidate_key=candidate_key,
            path=path,
            fatal=fatal,
            extra=extra or {},
        )

    @property
    def code(self) -> str:
        return self.failure.code

    def to_dict(self) -> dict[str, Any]:
        return self.failure.to_dict()


def sorted_failures(failures: list[Failure | BusinessError]) -> list[dict[str, Any]]:
    records = [entry.failure if isinstance(entry, BusinessError) else entry for entry in failures]
    return [record.to_dict() for record in sorted(records, key=Failure.sort_key)]
