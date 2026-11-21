"""Stable public error types and machine-readable error codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class Violation:
    shaper_id: str
    field: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "shaper_id": self.shaper_id,
            "field": self.field,
            "message": self.message,
        }


@dataclass
class PolicyError(ValueError):
    violations: list[Violation] = field(default_factory=list)
    code: str = "E_POLICY"

    def __post_init__(self) -> None:
        self.violations = sorted(
            self.violations,
            key=lambda item: (item.shaper_id, item.field, item.message),
        )
        message = f"{self.code}: " + "; ".join(
            f"{v.shaper_id}.{v.field}: {v.message}" for v in self.violations
        )
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "violations": [violation.as_dict() for violation in self.violations],
        }


class LedgerError(RuntimeError):
    code = "E_LEDGER"


class SnapshotError(RuntimeError):
    code = "E_SNAPSHOT"


@dataclass
class ReplayMismatch:
    packet_id: str
    shaper_id: str
    field: str
    expected: Any
    actual: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "shaper_id": self.shaper_id,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
        }


class ReplayError(RuntimeError):
    code = "E_REPLAY"

    def __init__(
        self,
        reason: str,
        *,
        basis: str,
        mismatch: ReplayMismatch | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.reason = reason
        self.basis = basis
        self.mismatch = mismatch
        self.details = dict(details or {})
        if mismatch is not None:
            message = (
                f"E_REPLAY: {reason}; first mismatch packet={mismatch.packet_id} "
                f"shaper={mismatch.shaper_id} field={mismatch.field} "
                f"expected={mismatch.expected!r} actual={mismatch.actual!r}; {basis}"
            )
        else:
            message = f"E_REPLAY: {reason}; {basis}"
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "reason": self.reason,
            "basis": self.basis,
            "details": self.details,
        }
        if self.mismatch is not None:
            result["first_mismatch"] = self.mismatch.as_dict()
        return result


ETHICAL_INGRESS_MARKER = "<ingress>"


def aggregate_policy_errors(items: Iterable[Violation]) -> PolicyError:
    return PolicyError(list(items))
