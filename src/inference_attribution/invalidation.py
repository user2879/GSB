"""Dependency-chain invalidation diagnostics and replay assertions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .errors import DiagnosticError
from .jsonio import canonical_json
from .planner import BatchPlan, DispatchPlan
from .spec import WorkerSet, WorkerSpec


@dataclass(frozen=True)
class InvalidationReport:
    direct_batch_ids: tuple[str, ...]
    indirect_batch_ids: tuple[str, ...]
    direct_request_ids: tuple[str, ...]
    indirect_request_ids: tuple[str, ...]
    unchanged_batch_ids: tuple[str, ...]
    worker_changes: tuple[dict[str, Any], ...]
    batch_details: dict[str, dict[str, Any]]

    def invalid_batch_ids(self) -> set[str]:
        return set(self.direct_batch_ids) | set(self.indirect_batch_ids)

    def invalid_new_plan_batch_ids(self, new_plan: DispatchPlan) -> set[str]:
        invalid = self.invalid_batch_ids()
        return {batch.batch_id for batch in new_plan.batches if batch.batch_id in invalid}

    def to_dict(self) -> dict[str, Any]:
        details = {
            batch_id: self.batch_details[batch_id]
            for batch_id in sorted(
                self.batch_details,
                key=lambda value: (
                    self.batch_details[value]["worker_id"],
                    self.batch_details[value]["batch_seq"],
                    value,
                ),
            )
        }
        return {
            "direct_invalid_batches": [
                details[batch_id] for batch_id in self.direct_batch_ids
            ],
            "indirect_invalid_batches": [
                details[batch_id] for batch_id in self.indirect_batch_ids
            ],
            "direct_invalid_request_ids": list(self.direct_request_ids),
            "indirect_invalid_request_ids": list(self.indirect_request_ids),
            "unchanged_batch_ids": list(self.unchanged_batch_ids),
            "worker_changes": list(self.worker_changes),
        }


def _worker_change_fields(old: WorkerSpec | None, new: WorkerSpec | None) -> list[str]:
    if old is None:
        return ["presence"]
    if new is None:
        return ["presence"]
    fields: list[str] = []
    if old.seed != new.seed:
        fields.append("seed")
    if old.input_dim != new.input_dim:
        fields.append("input_dim")
    if old.hidden_dim != new.hidden_dim:
        fields.append("hidden_dim")
    if old.param_version_seq != new.param_version_seq:
        fields.append("param_version_seq")
    if old.batch_size != new.batch_size:
        fields.append("batch_size")
    if old.concurrency_quota != new.concurrency_quota:
        fields.append("concurrency_quota")
    return fields


def _batch_sort_key(batch: BatchPlan) -> tuple[str, int, str]:
    return (batch.worker_id, batch.batch_seq, batch.batch_id)


def diagnose_invalidation(
    old_workers: WorkerSet,
    old_plan: DispatchPlan,
    new_workers: WorkerSet,
    new_plan: DispatchPlan,
) -> InvalidationReport:
    old_worker_map = old_workers.by_id()
    new_worker_map = new_workers.by_id()
    old_batches = old_plan.by_id()
    new_batches = new_plan.by_id()
    all_batches = {**old_batches, **new_batches}
    worker_ids = sorted(set(old_worker_map) | set(new_worker_map))
    direct_batches: set[str] = set()
    change_rows: list[dict[str, Any]] = []

    for worker_id in worker_ids:
        old_worker = old_worker_map.get(worker_id)
        new_worker = new_worker_map.get(worker_id)
        fields = _worker_change_fields(old_worker, new_worker)
        if fields:
            change_rows.append({"worker_id": worker_id, "changed_fields": fields})
        old_worker_batches = {
            batch.batch_id
            for batch in old_plan.batches
            if batch.worker_id == worker_id
        }
        new_worker_batches = {
            batch.batch_id
            for batch in new_plan.batches
            if batch.worker_id == worker_id
        }
        parameter_or_plan_change = bool(
            set(fields)
            & {
                "presence",
                "seed",
                "input_dim",
                "hidden_dim",
                "param_version_seq",
                "batch_size",
            }
        )
        if parameter_or_plan_change:
            direct_batches.update(old_worker_batches)
            direct_batches.update(new_worker_batches)
            continue
        for batch_id in sorted(old_worker_batches | new_worker_batches):
            old_batch = old_batches.get(batch_id)
            new_batch = new_batches.get(batch_id)
            if old_batch is None or new_batch is None:
                direct_batches.add(batch_id)
                continue
            if (
                old_batch.plan_digest != new_batch.plan_digest
                or old_batch.param_digest != new_batch.param_digest
                or old_batch.request_binding_digest != new_batch.request_binding_digest
                or old_batch.request_ids != new_batch.request_ids
                or old_batch.arrival_seqs != new_batch.arrival_seqs
            ):
                direct_batches.add(batch_id)

    indirect_requests: set[str] = set()
    for batch_id in direct_batches:
        batch = all_batches.get(batch_id)
        if batch is not None:
            indirect_requests.update(batch.request_ids)
    indirect_batches: set[str] = set()
    for batch in all_batches.values():
        if batch.batch_id in direct_batches:
            continue
        if indirect_requests.intersection(batch.request_ids):
            indirect_batches.add(batch.batch_id)
            indirect_requests.update(batch.request_ids)
    direct_requests: set[str] = set()
    unchanged = sorted(
        (
            batch.batch_id
            for batch in new_plan.batches
            if batch.batch_id not in direct_batches
            and batch.batch_id not in indirect_batches
        ),
        key=lambda batch_id: _batch_sort_key(new_batches[batch_id]),
    )
    details: dict[str, dict[str, Any]] = {}
    for batch_id, batch in all_batches.items():
        details[batch_id] = {
            "batch_id": batch_id,
            "worker_id": batch.worker_id,
            "batch_seq": batch.batch_seq,
            "request_ids": list(batch.request_ids),
            "present_in_old_plan": batch_id in old_batches,
            "present_in_new_plan": batch_id in new_batches,
        }
    direct_sorted = tuple(
        sorted(direct_batches, key=lambda batch_id: _batch_sort_key(all_batches[batch_id]))
    )
    indirect_sorted = tuple(
        sorted(
            indirect_batches,
            key=lambda batch_id: _batch_sort_key(all_batches[batch_id]),
        )
    )
    return InvalidationReport(
        direct_batch_ids=direct_sorted,
        indirect_batch_ids=indirect_sorted,
        direct_request_ids=tuple(sorted(direct_requests)),
        indirect_request_ids=tuple(sorted(indirect_requests)),
        unchanged_batch_ids=tuple(unchanged),
        worker_changes=tuple(change_rows),
        batch_details=details,
    )


def trace_for_batch(
    report: InvalidationReport,
    batch_id: str,
    *,
    mode: str = "recompute",
) -> dict[str, Any]:
    direct = batch_id in set(report.direct_batch_ids)
    indirect = batch_id in set(report.indirect_batch_ids)
    return {
        "mode": mode,
        "direct": direct,
        "indirect": indirect,
        "worker_changes": list(report.worker_changes),
    }


def _without_volatile_latency(row: dict[str, Any]) -> dict[str, Any]:
    projected = dict(row)
    receipts = []
    for receipt in row.get("normalized_receipts", []):
        item = dict(receipt)
        item.pop("latency_ms", None)
        receipts.append(item)
    projected["normalized_receipts"] = receipts
    return projected


REPLAY_FIELDS = (
    "status",
    "worker_id",
    "batch_seq",
    "param_version_seq",
    "param_digest",
    "request_ids",
    "arrival_seqs",
    "request_binding_digest",
    "plan_digest",
    "receipt_digest",
    "normalized_receipts",
    "invalidation_trace",
)


def assert_replay_consistent(
    plan: DispatchPlan | None,
    expected_rows: Iterable[dict[str, Any]],
    actual_rows: Iterable[dict[str, Any]],
) -> None:
    expected = {row["batch_id"]: row for row in expected_rows}
    actual = {row["batch_id"]: row for row in actual_rows}
    order = [batch.batch_id for batch in plan.sorted_batches()] if plan else []
    order.extend(sorted(set(expected) - set(order)))
    order.extend(sorted(set(actual) - set(order)))
    for batch_id in order:
        left = expected.get(batch_id)
        right = actual.get(batch_id)
        if left is None:
            raise DiagnosticError(
                code="E_REPLAY",
                message="restored ledger contains an unexpected batch",
                object=batch_id,
                field="presence",
            )
        if right is None:
            raise DiagnosticError(
                code="E_REPLAY",
                message="restored ledger is missing a planned batch",
                object=batch_id,
                field="presence",
            )
        left_view = _without_volatile_latency(left)
        right_view = _without_volatile_latency(right)
        for field in REPLAY_FIELDS:
            if canonical_json(left_view.get(field)) != canonical_json(
                right_view.get(field)
            ):
                raise DiagnosticError(
                    code="E_REPLAY",
                    message="replayed batch diverges from the reference ledger",
                    object=batch_id,
                    field=field,
                    details={
                        "expected": canonical_json(left_view.get(field)),
                        "actual": canonical_json(right_view.get(field)),
                    },
                )
