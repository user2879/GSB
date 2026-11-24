"""Dependency propagation from worker parameters to batches and ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .planning import BatchPlan, ExecutionPlan
from .specs import RequestSpec, SampleSpec, WorkerSpec


_WORKER_FIELDS = ("param_version", "seed", "model_dim", "hidden_dim", "batch_size")


@dataclass(frozen=True)
class InvalidationEntry:
    kind: str
    worker_id: str
    batch_id: str
    batch_seq: int
    old_batch_id: str | None
    request_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    dependency_path: tuple[str, ...]
    propagation: tuple[Mapping[str, Any], ...]
    direct: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "worker_id": self.worker_id,
            "batch_id": self.batch_id,
            "batch_seq": self.batch_seq,
            "old_batch_id": self.old_batch_id,
            "request_ids": list(self.request_ids),
            "reasons": list(self.reasons),
            "dependency_path": list(self.dependency_path),
            "propagation": [dict(step) for step in self.propagation],
            "direct": self.direct,
        }


@dataclass(frozen=True)
class InvalidationReport:
    rerun_entries: tuple[InvalidationEntry, ...]
    removed_entries: tuple[InvalidationEntry, ...]
    unaffected_batch_ids: tuple[str, ...]
    invalid_request_ids: tuple[str, ...]
    old_plan_fingerprint: str
    new_plan_fingerprint: str

    @property
    def rerun_batch_ids(self) -> tuple[str, ...]:
        return tuple(entry.batch_id for entry in self.rerun_entries)

    @property
    def removed_batch_ids(self) -> tuple[str, ...]:
        return tuple(entry.batch_id for entry in self.removed_entries)

    @property
    def unchanged_batch_ids(self) -> tuple[str, ...]:
        return self.unaffected_batch_ids

    @property
    def entry_by_batch_id(self) -> dict[str, InvalidationEntry]:
        return {entry.batch_id: entry for entry in self.rerun_entries}

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_plan_fingerprint": self.old_plan_fingerprint,
            "new_plan_fingerprint": self.new_plan_fingerprint,
            "rerun_batches": [entry.to_dict() for entry in self.rerun_entries],
            "removed_batches": [entry.to_dict() for entry in self.removed_entries],
            "unaffected_batch_ids": list(self.unaffected_batch_ids),
            "invalid_request_ids": list(self.invalid_request_ids),
        }


def _worker_changes(
    old_workers: Mapping[str, WorkerSpec], new_workers: Mapping[str, WorkerSpec]
) -> dict[str, list[tuple[str, Any, Any]]]:
    changes: dict[str, list[tuple[str, Any, Any]]] = {}
    for worker_id in sorted(set(old_workers) | set(new_workers)):
        old = old_workers.get(worker_id)
        new = new_workers.get(worker_id)
        if old is None:
            changes[worker_id] = [("worker_added", None, new.digest())]
            continue
        if new is None:
            changes[worker_id] = [("worker_removed", old.digest(), None)]
            continue
        field_changes: list[tuple[str, Any, Any]] = []
        for field_name in _WORKER_FIELDS:
            old_value = getattr(old, field_name)
            new_value = getattr(new, field_name)
            if old_value != new_value:
                field_changes.append((field_name, old_value, new_value))
        if field_changes:
            changes[worker_id] = field_changes
    return changes


def _request_changes(
    old_requests: Mapping[str, RequestSpec], new_requests: Mapping[str, RequestSpec]
) -> dict[str, tuple[str, ...]]:
    changed: dict[str, tuple[str, ...]] = {}
    for request_id in sorted(set(old_requests) | set(new_requests)):
        old = old_requests.get(request_id)
        new = new_requests.get(request_id)
        if old is None:
            changed[request_id] = ("request_added",)
            continue
        if new is None:
            changed[request_id] = ("request_removed",)
            continue
        fields: list[str] = []
        if old.arrival_seq != new.arrival_seq:
            fields.append("arrival_seq")
        if old.input_vector != new.input_vector:
            fields.append("input")
        if old.duplicate_refs != new.duplicate_refs:
            fields.append("duplicate_refs")
        if fields:
            changed[request_id] = tuple(f"request.{name}.changed" for name in fields)
    return changed


def _worker_steps(
    worker_id: str, changes: Iterable[tuple[str, Any, Any]]
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    reasons: list[str] = []
    steps: list[dict[str, Any]] = []
    path: list[str] = [f"worker:{worker_id}"]
    for field_name, old_value, new_value in changes:
        reason = f"worker.{field_name}.changed"
        reasons.append(reason)
        path.append(f"{field_name}:{old_value}->{new_value}")
        steps.append(
            {
                "from": f"worker:{worker_id}",
                "to": f"worker:{worker_id}#{field_name}",
                "edge": "worker_signature",
                "reason": f"{field_name} changed from {old_value!r} to {new_value!r}",
            }
        )
    return reasons, steps, path


def _batch_steps(
    worker_id: str,
    batch: BatchPlan,
    *,
    root_steps: Iterable[Mapping[str, Any]],
    root_path: Iterable[str],
    reasons: Iterable[str],
    direct: bool,
) -> InvalidationEntry:
    reason_list = list(reasons)
    steps = [dict(step) for step in root_steps]
    path = list(root_path)
    steps.append(
        {
            "from": f"worker:{worker_id}",
            "to": f"batch:{batch.batch_id}",
            "edge": "worker_parameter_version_to_batch",
            "reason": "every batch owned by the worker inherits invalid parameters",
        }
    )
    path.append(f"batch:{batch.batch_id}")
    for request_id in sorted(batch.request_ids):
        steps.append(
            {
                "from": f"batch:{batch.batch_id}",
                "to": f"request:{request_id}",
                "edge": "batch_to_request_ownership",
                "reason": "request ownership is derived from the invalidated batch",
            }
        )
        path.append(f"request:{request_id}")
    return InvalidationEntry(
        kind="rerun",
        worker_id=worker_id,
        batch_id=batch.batch_id,
        batch_seq=batch.batch_seq,
        old_batch_id=batch.batch_id,
        request_ids=tuple(sorted(batch.request_ids)),
        reasons=tuple(sorted(set(reason_list))),
        dependency_path=tuple(path),
        propagation=tuple(steps),
        direct=direct,
    )


def _removed_entry(
    batch: BatchPlan,
    *,
    reasons: Iterable[str],
) -> InvalidationEntry:
    reason_list = tuple(sorted(set(reasons)))
    path = [f"worker:{batch.worker_id}", f"batch:{batch.batch_id}"]
    propagation = [
        {
            "from": f"worker:{batch.worker_id}",
            "to": f"batch:{batch.batch_id}",
            "edge": "batch_lifecycle",
            "reason": "; ".join(reason_list),
        }
    ]
    for request_id in sorted(batch.request_ids):
        propagation.append(
            {
                "from": f"batch:{batch.batch_id}",
                "to": f"request:{request_id}",
                "edge": "batch_to_request_ownership",
                "reason": "ownership record is removed with its old batch",
            }
        )
        path.append(f"request:{request_id}")
    return InvalidationEntry(
        kind="removed",
        worker_id=batch.worker_id,
        batch_id=batch.batch_id,
        batch_seq=batch.batch_seq,
        old_batch_id=batch.batch_id,
        request_ids=tuple(sorted(batch.request_ids)),
        reasons=reason_list,
        dependency_path=tuple(path),
        propagation=tuple(propagation),
        direct="worker.batch_size.changed" in reason_list
        or "worker.worker_removed" in reason_list,
    )


def compute_invalidation(
    old_spec: SampleSpec,
    new_spec: SampleSpec,
    old_plan: ExecutionPlan,
    new_plan: ExecutionPlan,
) -> InvalidationReport:
    worker_changes = _worker_changes(old_spec.worker_by_id, new_spec.worker_by_id)
    request_changes = _request_changes(old_spec.request_by_id, new_spec.request_by_id)
    old_by_id = old_plan.by_id
    new_by_id = new_plan.by_id
    rerun: list[InvalidationEntry] = []
    removed: list[InvalidationEntry] = []

    for batch in new_plan.sorted_batches():
        reasons: list[str] = []
        root_steps: list[dict[str, Any]] = []
        root_path: list[str] = []
        direct = False

        changes = worker_changes.get(batch.worker_id)
        if changes:
            worker_reasons, steps, path = _worker_steps(batch.worker_id, changes)
            reasons.extend(worker_reasons)
            root_steps.extend(steps)
            root_path.extend(path)
            direct = True

        if batch.batch_id not in old_by_id:
            reasons.append("batch.partition_or_routing.changed")
            root_steps.append(
                {
                    "from": f"worker:{batch.worker_id}",
                    "to": f"batch:{batch.batch_id}",
                    "edge": "deterministic_batch_plan",
                    "reason": "batch identity has no matching predecessor",
                }
            )
            root_path.extend([f"worker:{batch.worker_id}", f"batch:{batch.batch_id}"])
            direct = True
        else:
            old_batch = old_by_id[batch.batch_id]
            if old_batch.request_ids != batch.request_ids:
                reasons.append("batch.membership.changed")
                direct = True

        request_reasons: list[str] = []
        for request_id in sorted(batch.request_ids):
            changes_for_request = request_changes.get(request_id)
            if changes_for_request:
                request_reasons.extend(changes_for_request)
                root_steps.append(
                    {
                        "from": f"request:{request_id}",
                        "to": f"batch:{batch.batch_id}",
                        "edge": "request_payload_to_batch",
                        "reason": ", ".join(changes_for_request),
                    }
                )
                root_path.extend([f"request:{request_id}", f"batch:{batch.batch_id}"])
        if request_reasons:
            reasons.extend(request_reasons)
            direct = True

        if reasons:
            rerun.append(
                _batch_steps(
                    batch.worker_id,
                    batch,
                    root_steps=root_steps,
                    root_path=root_path,
                    reasons=reasons,
                    direct=direct,
                )
            )

    rerun_ids = {entry.batch_id for entry in rerun}
    for batch in old_plan.sorted_batches():
        if batch.batch_id in new_by_id:
            continue
        removal_reasons = ["batch.no_longer_in_plan"]
        if batch.worker_id in worker_changes:
            removal_reasons.extend(
                f"worker.{field_name}.changed"
                for field_name, _, _ in worker_changes[batch.worker_id]
            )
        removed.append(_removed_entry(batch, reasons=removal_reasons))

    rerun.sort(key=lambda entry: (entry.worker_id, entry.batch_seq, entry.batch_id))
    removed.sort(key=lambda entry: (entry.worker_id, entry.batch_seq, entry.batch_id))
    invalid_requests = sorted(
        {
            request_id
            for entry in rerun
            for request_id in entry.request_ids
        }
        | {
            request_id
            for request_id, reasons in request_changes.items()
            if "request_removed" not in reasons
        }
    )
    unaffected = tuple(
        entry.batch_id
        for entry in new_plan.sorted_batches()
        if entry.batch_id not in rerun_ids
    )
    return InvalidationReport(
        rerun_entries=tuple(rerun),
        removed_entries=tuple(removed),
        unaffected_batch_ids=unaffected,
        invalid_request_ids=tuple(invalid_requests),
        old_plan_fingerprint=old_plan.fingerprint,
        new_plan_fingerprint=new_plan.fingerprint,
    )
