from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic_ns
from typing import Iterable, Sequence

from .derive import resolve_effective
from .engine import JudgmentEngine
from .errors import AppError, E_LATENCY, E_PREVIEW, PreviewError
from .logging_utils import StructuredLogger
from .models import (
    AccessEvent,
    ConfigDocument,
    DECISION_ALLOW,
    DECISION_OVER_QUOTA,
    DECISION_RATE_LIMITED,
    Judgment,
    VolumeUsage,
    judgment_map,
)


@dataclass
class PreviewResult:
    judgments: list[Judgment]
    usage: dict[str, VolumeUsage]
    elapsed_ms: int
    budget_ms: int
    error_bound_single: int = 0
    error_bound_aggregate: int = 0
    decision_counts: dict[str, int] = field(default_factory=dict)
    latency_exceeded: bool = False

    def as_dict(self) -> dict:
        return {
            "elapsed_ms": self.elapsed_ms,
            "budget_ms": self.budget_ms,
            "latency_exceeded": self.latency_exceeded,
            "error_bound_single": self.error_bound_single,
            "error_bound_aggregate": self.error_bound_aggregate,
            "decision_counts": dict(self.decision_counts),
            "judgments": [judgment.to_json() for judgment in self.judgments],
            "usage": {
                volume_id: self.usage[volume_id].to_json()
                for volume_id in sorted(self.usage)
            },
        }


@dataclass
class PreviewComparison:
    elapsed_ms: int
    budget_ms: int
    mismatches: list[dict] = field(default_factory=list)
    errors: list[AppError] = field(default_factory=list)
    latency_exceeded: bool = False

    @property
    def ok(self) -> bool:
        return not self.mismatches and not self.errors

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "elapsed_ms": self.elapsed_ms,
            "budget_ms": self.budget_ms,
            "latency_exceeded": self.latency_exceeded,
            "mismatches": self.mismatches,
            "errors": [error.as_dict() for error in self.errors],
        }


class PreviewService:
    """Fast interactive preview plus an independent authoritative replay."""

    def __init__(
        self,
        events: Iterable[AccessEvent],
        *,
        logger: StructuredLogger | None = None,
        budget_ms: int = 50,
    ) -> None:
        if budget_ms < 0:
            raise ValueError("budget_ms must be non-negative")
        self.events = list(events)
        self.logger = logger
        self.budget_ms = budget_ms

    def fast_preview(self, document: ConfigDocument) -> PreviewResult:
        return self._evaluate(document, operation="preview.fast")

    def authoritative_replay(self, document: ConfigDocument) -> PreviewResult:
        return self._evaluate(document, operation="preview.authoritative")

    def _evaluate(self, document: ConfigDocument, *, operation: str) -> PreviewResult:
        started = monotonic_ns()
        effective = resolve_effective(document)
        engine = JudgmentEngine(effective)
        judgments = [engine.evaluate(event) for event in self.events]
        elapsed_ms = (monotonic_ns() - started) // 1_000_000
        counts = {
            DECISION_ALLOW: 0,
            DECISION_RATE_LIMITED: 0,
            DECISION_OVER_QUOTA: 0,
        }
        for judgment in judgments:
            counts[judgment.decision] += 1
        exceeded = elapsed_ms > self.budget_ms
        if exceeded and self.logger is not None:
            self.logger.emit(
                E_LATENCY,
                "preview exceeded the explicit integer millisecond budget",
                operation=operation,
                details={
                    "elapsed_ms": elapsed_ms,
                    "budget_ms": self.budget_ms,
                    "event_count": len(self.events),
                    "non_blocking": True,
                },
            )
        return PreviewResult(
            judgments=judgments,
            usage=engine.snapshot_usage(),
            elapsed_ms=elapsed_ms,
            budget_ms=self.budget_ms,
            error_bound_single=0,
            error_bound_aggregate=0,
            decision_counts=counts,
            latency_exceeded=exceeded,
        )

    def compare(self, preview: PreviewResult, authoritative: PreviewResult) -> PreviewComparison:
        started = monotonic_ns()
        left = judgment_map(preview.judgments)
        right = judgment_map(authoritative.judgments)
        mismatches: list[dict] = []
        errors: list[AppError] = []
        for seq in sorted(set(left) | set(right)):
            first = left.get(seq)
            second = right.get(seq)
            if first is None or second is None:
                available = second or first
                assert available is not None
                mismatch = {
                    "volume_id": available.volume_id,
                    "seq": seq,
                    "difference_fields": [
                        "missing_preview_judgment" if first is None else "missing_authoritative_judgment"
                    ],
                }
            else:
                fields = first.difference_fields(second)
                if not fields:
                    continue
                mismatch = {
                    "volume_id": first.volume_id,
                    "seq": seq,
                    "difference_fields": list(fields),
                    "preview": first.comparable(),
                    "authoritative": second.comparable(),
                }
            mismatches.append(mismatch)
            error = PreviewError(
                "interactive preview differs from authoritative replay",
                volume_id=mismatch["volume_id"],
                field_path=f"event:{seq}",
                details=mismatch,
            )
            errors.append(error)
            if self.logger is not None:
                self.logger.emit(
                    E_PREVIEW,
                    error.message,
                    operation="preview.compare",
                    volume_id=error.volume_id,
                    field_path=error.field_path,
                    details=mismatch,
                )

        elapsed_ms = (monotonic_ns() - started) // 1_000_000
        exceeded = (
            preview.latency_exceeded
            or authoritative.latency_exceeded
            or elapsed_ms > self.budget_ms
        )
        if elapsed_ms > self.budget_ms and self.logger is not None:
            self.logger.emit(
                E_LATENCY,
                "preview comparison exceeded the explicit integer millisecond budget",
                operation="preview.compare",
                details={
                    "elapsed_ms": elapsed_ms,
                    "budget_ms": self.budget_ms,
                    "non_blocking": True,
                },
            )
        return PreviewComparison(
            elapsed_ms=elapsed_ms,
            budget_ms=self.budget_ms,
            mismatches=mismatches,
            errors=errors,
            latency_exceeded=exceeded,
        )

    def run(self, document: ConfigDocument) -> tuple[PreviewResult, PreviewResult, PreviewComparison]:
        preview = self.fast_preview(document)
        authoritative = self.authoritative_replay(document)
        comparison = self.compare(preview, authoritative)
        return preview, authoritative, comparison


def replay_from_ledger_events(
    document: ConfigDocument,
    events: Sequence[AccessEvent],
) -> PreviewResult:
    """Small explicit helper used by the GUI worker thread."""

    service = PreviewService(events, budget_ms=0)
    return service.authoritative_replay(document)
