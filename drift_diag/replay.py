"""Snapshot-based deterministic replay verification against the ledger."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .canonical import canonical_json
from .config import EvaluationConfig
from .drift import aggregate_all, compare_field
from .errors import replay_error
from .events import apply_event
from .ledger import LedgerStore
from .models import FIELD_INDEX, FIELD_ORDER, ObservationEvent, TargetSet


DECISION_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("target", "target"),
    ("observed", "observed"),
    ("observed_present", "observed"),
    ("drift", "drift"),
    ("reason", "reason"),
    ("basis", "basis"),
    ("fixed_metric", "metric"),
    ("field_version", "field_version"),
)


@dataclass
class ReplayResult:
    ok: bool
    batches_checked: int
    decisions_checked: int
    latest_event_cursor: int | None
    snapshot_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "batches_checked": self.batches_checked,
            "decisions_checked": self.decisions_checked,
            "latest_event_cursor": self.latest_event_cursor,
            "snapshot_digest": self.snapshot_digest,
        }


def verify_replay(
    ledger: LedgerStore,
    target: TargetSet,
    events: list[ObservationEvent],
    config: EvaluationConfig | None = None,
) -> ReplayResult:
    config = (config or EvaluationConfig()).validate()
    integrity = ledger.verify_integrity()
    snapshot = ledger.read_snapshot()
    if snapshot is None:
        raise replay_error("no snapshot is available for replay", field="<snapshot>")
    if snapshot["target_digest"] != target.target_digest:
        raise replay_error(
            "snapshot target differs from the reconstructed target",
            item="snapshot",
            field="target_digest",
            details={"expected": snapshot["target_digest"], "actual": target.target_digest},
        )

    batches = ledger.read_batches()
    actual_state: dict[str, dict[str, Any]] = {}
    previous_cursor = 0
    decisions_checked = 0
    for batch in batches:
        cursor = batch["event_cursor"]
        if cursor < previous_cursor or cursor > len(events):
            raise replay_error(
                "non-monotonic or out-of-range event cursor",
                item=batch["batch_key"],
                field="event_cursor",
                details={"previous": previous_cursor, "actual": cursor},
            )
        for event in events[previous_cursor:cursor]:
            apply_event(actual_state, event)

        recomputed = [
            compare_field(
                target.workloads[workload.name],
                field,
                actual_state,
                scale_exp=batch["scale_exp"],
            )
            for workload in target.ordered_workloads()
            for field in FIELD_ORDER
        ]
        _compare_decisions(batch, recomputed)
        decisions_checked += len(recomputed)

        stored_metrics = batch["workload_metrics"]
        tolerance = _tolerance_from_metrics(stored_metrics, config)
        batch_config = EvaluationConfig(
            scale_exp=batch["scale_exp"],
            scale_version=batch["scale_version"],
            tolerance_num=tolerance[0],
            tolerance_den=tolerance[1],
            global_limit=config.global_limit,
            per_workload_quota=config.per_workload_quota,
            task_delay_ms=config.task_delay_ms,
            batch_events=config.batch_events,
        ).validate()
        recomputed_metrics = aggregate_all(target, recomputed, batch_config)
        _compare_metrics(batch, stored_metrics, recomputed_metrics)
        previous_cursor = cursor

    if previous_cursor != len(events) and not batches:
        raise replay_error("ledger contains no batches to replay", field="<ledger>")
    return ReplayResult(
        ok=True,
        batches_checked=len(batches),
        decisions_checked=decisions_checked,
        latest_event_cursor=previous_cursor if batches else None,
        snapshot_digest=snapshot["snapshot_digest"],
    )


def _tolerance_from_metrics(metrics: dict[str, Any], config: EvaluationConfig) -> tuple[int, int]:
    for workload in metrics.get("workloads", {}).values():
        tolerance = workload.get("tolerance")
        if tolerance:
            return int(tolerance["numerator"]), int(tolerance["denominator"])
    return config.tolerance_num, config.tolerance_den


def _compare_decisions(batch: dict[str, Any], recomputed: list[dict[str, Any]]) -> None:
    stored = {
        (decision["workload"], decision["field"]): decision
        for decision in batch["decisions"]
    }
    expected = {
        (decision["workload"], decision["field"]): decision
        for decision in recomputed
    }
    all_keys = sorted(
        set(stored) | set(expected), key=lambda key: (key[0], FIELD_INDEX[key[1]])
    )
    for key in all_keys:
        workload, field = key
        if key not in stored or key not in expected:
            raise replay_error(
                "replayed decision membership differs",
                item=workload,
                field=field,
                details={
                    "category": "membership",
                    "batch_key": batch["batch_key"],
                    "event_cursor": batch["event_cursor"],
                    "stored_present": key in stored,
                    "replayed_present": key in expected,
                },
            )
        left, right = stored[key], expected[key]
        for key_name, category in DECISION_CATEGORIES:
            if canonical_json(left[key_name]) != canonical_json(right[key_name]):
                raise replay_error(
                    "replayed drift decision differs from the ledger",
                    item=workload,
                    field=field,
                    details={
                        "category": category,
                        "batch_key": batch["batch_key"],
                        "event_cursor": batch["event_cursor"],
                        "expected": deepcopy(left[key_name]),
                        "replayed": deepcopy(right[key_name]),
                    },
                )


def _compare_metrics(
    batch: dict[str, Any],
    stored: dict[str, Any],
    recomputed: dict[str, Any],
) -> None:
    if canonical_json(stored) == canonical_json(recomputed):
        return
    if stored.get("overall_converged") != recomputed.get("overall_converged"):
        raise replay_error(
            "replayed overall convergence differs from the ledger",
            item="<all>",
            field="<overall_converged>",
            details={
                "category": "convergence",
                "batch_key": batch["batch_key"],
                "expected": stored.get("overall_converged"),
                "replayed": recomputed.get("overall_converged"),
            },
        )
    stored_workloads = stored.get("workloads", {})
    recomputed_workloads = recomputed.get("workloads", {})
    for workload in sorted(set(stored_workloads) | set(recomputed_workloads)):
        left = stored_workloads.get(workload)
        right = recomputed_workloads.get(workload)
        if left is None or right is None:
            raise replay_error(
                "replayed workload metric membership differs",
                item=workload,
                field="<workload_metrics>",
                details={
                    "category": "membership",
                    "batch_key": batch["batch_key"],
                    "stored_present": left is not None,
                    "replayed_present": right is not None,
                },
            )
        for key in sorted(set(left) | set(right)):
            if canonical_json(left.get(key)) != canonical_json(right.get(key)):
                category = "convergence" if key in {"converged", "boundary_sensitive"} else "metric"
                raise replay_error(
                    "replayed workload metric differs from the ledger",
                    item=workload,
                    field=key,
                    details={
                        "category": category,
                        "batch_key": batch["batch_key"],
                        "expected": left.get(key),
                        "replayed": right.get(key),
                    },
                )
    raise replay_error(
        "replayed workload metrics differ from the ledger",
        item="<all>",
        field="<workload_metrics>",
        details={"category": "metric", "batch_key": batch["batch_key"]},
    )
