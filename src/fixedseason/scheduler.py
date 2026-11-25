"""Deterministic budget planning and batch construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import E_FIT_BUDGET_EXCEEDED, E_POINT_BUDGET_PENDING, Issue
from .model import FitConfig, SeriesRecord


@dataclass(frozen=True)
class PlanEntry:
    series_id: str
    observation_points: int
    kind: str
    required_points: int
    deducted_points: int
    remaining_before: int
    remaining_after: int
    issue: Issue | None = None
    batch_index: int | None = None

    def ledger_dict(self, execution_outcome: str | None = None) -> dict[str, Any]:
        data = {
            "series_id": self.series_id,
            "observation_points": self.observation_points,
            "required_points": self.required_points,
            "deducted_points": self.deducted_points,
            "remaining_before": self.remaining_before,
            "remaining_after": self.remaining_after,
            "outcome": execution_outcome or self.kind,
        }
        if self.issue is not None:
            data["code"] = self.issue.code
            data["reason"] = self.issue.message
        return data

    def as_manifest_dict(self) -> dict[str, Any]:
        data = self.ledger_dict()
        data["kind"] = self.kind
        if self.issue is not None:
            data["issue"] = self.issue.as_dict()
        if self.batch_index is not None:
            data["batch_index"] = self.batch_index
        return data


@dataclass(frozen=True)
class BudgetPlan:
    entries: tuple[PlanEntry, ...]
    batches: tuple[tuple[str, ...], ...]
    consumed_points: int
    remaining_points: int
    pending_queue: tuple[Issue, ...]
    failures: tuple[Issue, ...]
    cache_hits: tuple[str, ...]

    @property
    def fit_series_ids(self) -> tuple[str, ...]:
        return tuple(entry.series_id for entry in self.entries if entry.kind == "fit")

    def entry_map(self) -> dict[str, PlanEntry]:
        return {entry.series_id: entry for entry in self.entries}

    def ledger(self, execution_outcomes: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
        outcomes = execution_outcomes or {}
        return [
            entry.ledger_dict(outcomes.get(entry.series_id))
            for entry in self.entries
        ]


def build_budget_plan(
    series: tuple[SeriesRecord, ...],
    config: FitConfig,
    cached_series_ids: set[str],
    *,
    prior_plan: BudgetPlan | None = None,
) -> BudgetPlan:
    """Plan in series-id order and reserve budget before any worker starts."""

    if prior_plan is not None:
        return prior_plan
    remaining = int(config.global_fit_points)
    entries: list[PlanEntry] = []
    pending: list[Issue] = []
    failures: list[Issue] = []
    cache_hits: list[str] = []
    fit_ids: list[str] = []

    for record in sorted(series, key=lambda item: item.series_id):
        points = record.length
        before = remaining
        if record.series_id in cached_series_ids:
            cache_hits.append(record.series_id)
            entries.append(
                PlanEntry(
                    series_id=record.series_id,
                    observation_points=points,
                    kind="cache_hit",
                    required_points=0,
                    deducted_points=0,
                    remaining_before=before,
                    remaining_after=remaining,
                )
            )
            continue
        if points > config.per_series_quota:
            issue = Issue(
                code=E_FIT_BUDGET_EXCEEDED,
                message=(
                    f"series requires {points} observation points, exceeding "
                    f"per_series_quota {config.per_series_quota}"
                ),
                series_id=record.series_id,
                stage="budget",
            )
            failures.append(issue)
            entries.append(
                PlanEntry(
                    series_id=record.series_id,
                    observation_points=points,
                    kind="budget_exceeded",
                    required_points=points,
                    deducted_points=0,
                    remaining_before=before,
                    remaining_after=remaining,
                    issue=issue,
                )
            )
            continue
        if points > config.batch_observation_points:
            issue = Issue(
                code=E_POINT_BUDGET_PENDING,
                message=(
                    f"series observation count {points} cannot fit in one batch of "
                    f"{config.batch_observation_points} points"
                ),
                series_id=record.series_id,
                stage="budget",
            )
            pending.append(issue)
            entries.append(
                PlanEntry(
                    series_id=record.series_id,
                    observation_points=points,
                    kind="pending",
                    required_points=points,
                    deducted_points=0,
                    remaining_before=before,
                    remaining_after=remaining,
                    issue=issue,
                )
            )
            continue
        if points > remaining:
            issue = Issue(
                code=E_POINT_BUDGET_PENDING,
                message=(
                    f"global fit budget exhausted: need {points}, remaining {remaining}"
                ),
                series_id=record.series_id,
                stage="budget",
            )
            pending.append(issue)
            entries.append(
                PlanEntry(
                    series_id=record.series_id,
                    observation_points=points,
                    kind="pending",
                    required_points=points,
                    deducted_points=0,
                    remaining_before=before,
                    remaining_after=remaining,
                    issue=issue,
                )
            )
            continue
        remaining -= points
        fit_ids.append(record.series_id)
        entries.append(
            PlanEntry(
                series_id=record.series_id,
                observation_points=points,
                kind="fit",
                required_points=points,
                deducted_points=points,
                remaining_before=before,
                remaining_after=remaining,
            )
        )

    lengths = {record.series_id: record.length for record in series}
    batches: list[tuple[str, ...]] = []
    current: list[str] = []
    current_points = 0
    fit_batch_index: dict[str, int] = {}
    for series_id in fit_ids:
        points = lengths[series_id]
        if current and current_points + points > config.batch_observation_points:
            batches.append(tuple(current))
            current = []
            current_points = 0
        if not current:
            fit_batch_index[series_id] = len(batches)
        current.append(series_id)
        current_points += points
    if current:
        batches.append(tuple(current))

    entries = [
        PlanEntry(
            series_id=entry.series_id,
            observation_points=entry.observation_points,
            kind=entry.kind,
            required_points=entry.required_points,
            deducted_points=entry.deducted_points,
            remaining_before=entry.remaining_before,
            remaining_after=entry.remaining_after,
            issue=entry.issue,
            batch_index=fit_batch_index.get(entry.series_id),
        )
        for entry in entries
    ]
    return BudgetPlan(
        entries=tuple(entries),
        batches=tuple(batches),
        consumed_points=sum(entry.deducted_points for entry in entries),
        remaining_points=remaining,
        pending_queue=tuple(pending),
        failures=tuple(failures),
        cache_hits=tuple(cache_hits),
    )
