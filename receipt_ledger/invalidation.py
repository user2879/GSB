from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .models import Plan, SpecBundle, WorkerSpec


PARAMETER_FIELDS = ("seed", "model_dim", "hidden_dim", "param_version")
DEPENDENCY_FIELDS = ("seed", "model_dim", "hidden_dim", "param_version", "batch_size")


@dataclass
class InvalidationReport:
    source_changes: list[dict[str, Any]] = field(default_factory=list)
    batch_nodes: list[dict[str, Any]] = field(default_factory=list)
    request_nodes: list[dict[str, Any]] = field(default_factory=list)
    recompute_batch_ids: list[str] = field(default_factory=list)
    recompute_request_ids: list[str] = field(default_factory=list)
    reattribute_request_ids: list[str] = field(default_factory=list)
    reusable_batch_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def invalid_batch_ids(self) -> list[str]:
        return _unique([node["batch_id"] for node in self.batch_nodes])

    @property
    def invalid_request_ids(self) -> list[str]:
        return _unique([node["request_id"] for node in self.request_nodes])

    @property
    def direct_batch_ids(self) -> list[str]:
        return _unique([node["batch_id"] for node in self.batch_nodes if node.get("dependency_level") == "direct"])

    @property
    def indirect_batch_ids(self) -> list[str]:
        return _unique([node["batch_id"] for node in self.batch_nodes if node.get("dependency_level") == "indirect"])

    def propagation_rows(self, epoch: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for source in self.source_changes:
            rows.append(
                {
                    "epoch": epoch,
                    "worker_id": source["worker_id"],
                    "batch_id": None,
                    "request_id": None,
                    "node_type": "worker",
                    "dependency_level": "source",
                    "cause": "worker_spec_changed",
                    "basis": source,
                }
            )
        for batch in self.batch_nodes:
            rows.append(
                {
                    "epoch": epoch,
                    "worker_id": batch["worker_id"],
                    "batch_id": batch["batch_id"],
                    "request_id": None,
                    "node_type": "batch",
                    "dependency_level": batch["dependency_level"],
                    "cause": ",".join(batch["causes"]),
                    "basis": batch,
                }
            )
        for request in self.request_nodes:
            rows.append(
                {
                    "epoch": epoch,
                    "worker_id": request["worker_id"],
                    "batch_id": request.get("batch_id"),
                    "request_id": request["request_id"],
                    "node_type": "request_attribution",
                    "dependency_level": request.get("dependency_level", "indirect"),
                    "cause": ",".join(request["causes"]),
                    "basis": request,
                }
            )
        return rows


def _worker_changes(old: WorkerSpec | None, new: WorkerSpec | None) -> dict[str, Any] | None:
    worker_id = (new or old).worker_id  # type: ignore[union-attr]
    if old is None and new is not None:
        return {"worker_id": worker_id, "change": "added", "changed_fields": list(DEPENDENCY_FIELDS)}
    if new is None and old is not None:
        return {"worker_id": worker_id, "change": "removed", "changed_fields": list(DEPENDENCY_FIELDS)}
    assert old is not None and new is not None
    changed: dict[str, dict[str, Any]] = {}
    for field_name in DEPENDENCY_FIELDS:
        old_value = getattr(old, field_name)
        new_value = getattr(new, field_name)
        if old_value != new_value:
            changed[field_name] = {"old": old_value, "new": new_value}
    if not changed:
        return None
    return {
        "worker_id": worker_id,
        "change": "modified",
        "changed_fields": sorted(changed),
        "values": changed,
        "parameter_changed": any(name in changed for name in PARAMETER_FIELDS),
        "batch_size_changed": "batch_size" in changed,
    }


def propagate_changes(
    old_bundle: SpecBundle | None,
    new_bundle: SpecBundle,
    old_plan: Plan | None,
    new_plan: Plan,
) -> InvalidationReport:
    report = InvalidationReport()
    old_workers = old_bundle.worker_map() if old_bundle else {}
    new_workers = new_bundle.worker_map()
    old_batch_map = old_plan.batch_map() if old_plan else {}
    new_batch_map = new_plan.batch_map()
    old_assignment = _assignment_map(old_plan)
    new_assignment = _assignment_map(new_plan)

    old_membership = {
        (batch.worker_id, batch.arrival_seqs): batch.batch_id
        for batch in (old_plan.batches if old_plan else ())
    }

    worker_ids = sorted(set(old_workers) | set(new_workers))
    changes_by_worker: dict[str, dict[str, Any]] = {}
    for worker_id in worker_ids:
        change = _worker_changes(old_workers.get(worker_id), new_workers.get(worker_id))
        if change:
            report.source_changes.append(change)
            changes_by_worker[worker_id] = change

    old_batches_by_worker: dict[str, list[Any]] = {}
    for batch in sorted(old_plan.batches if old_plan else (), key=lambda item: (item.worker_id, item.batch_index)):
        old_batches_by_worker.setdefault(batch.worker_id, []).append(batch)

    for worker_id in worker_ids:
        change = changes_by_worker.get(worker_id, {})
        parameter_changed = bool(change.get("parameter_changed")) or change.get("change") in {"added", "removed"}
        batch_size_changed = bool(change.get("batch_size_changed"))
        direct_worker_dependency = parameter_changed or batch_size_changed or bool(change)
        for old_batch in old_batches_by_worker.get(worker_id, []):
            causes: list[str] = []
            if parameter_changed:
                fields = change.get("changed_fields", PARAMETER_FIELDS)
                param_fields = [name for name in fields if name in PARAMETER_FIELDS] or list(PARAMETER_FIELDS)
                causes.append("worker_parameter_dependency:" + "+".join(param_fields))
            membership_key = (old_batch.worker_id, old_batch.arrival_seqs)
            membership_exists = membership_key in {
                (batch.worker_id, batch.arrival_seqs) for batch in new_plan.batches
            }
            if not membership_exists:
                causes.append("batch_size_or_stream_membership_changed")
            moved_requests = []
            for request_id in old_batch.request_ids:
                old_loc = old_assignment.get(request_id)
                new_loc = new_assignment.get(request_id)
                if old_loc != new_loc:
                    moved_requests.append(request_id)
            if moved_requests and membership_exists:
                causes.append("request_attribution_changed")
            if not causes:
                continue
            dependency_level = "direct" if direct_worker_dependency else "indirect"

            report.batch_nodes.append(
                {
                    "batch_id": old_batch.batch_id,
                    "worker_id": old_batch.worker_id,
                    "batch_index": old_batch.batch_index,
                    "identity": "old_frozen_batch",
                    "dependency_level": dependency_level,
                    "causes": causes,
                    "basis": {
                        "dependency_chain": "worker_spec -> batch_summary -> request_attribution",
                        "worker_change": change or {"changed_fields": []},
                        "membership_exists_in_new_plan": membership_exists,
                        "moved_request_ids": moved_requests,
                        "frozen_digest_policy": "retain_immutably",
                    },
                }
            )

            for rank, request_id in enumerate(old_batch.request_ids):
                request_causes: list[str] = []
                old_loc = old_assignment.get(request_id)
                new_loc = new_assignment.get(request_id)
                if parameter_changed:
                    request_causes.append("summary_basis_invalidated_by_parameter_version")
                if old_loc != new_loc:
                    request_causes.append("attribution_boundary_changed")
                if new_loc is None:
                    request_causes.append("request_absent_from_new_stream")
                if request_causes:
                    report.request_nodes.append(
                        {
                            "request_id": request_id,
                            "worker_id": old_batch.worker_id,
                            "batch_id": old_batch.batch_id,
                            "rank": rank,
                            "causes": request_causes,
                            "old_assignment": old_loc,
                            "new_assignment": new_loc,
                        }
                    )

    for new_batch in sorted(new_plan.batches, key=lambda item: (item.worker_id, item.batch_index)):
        change = changes_by_worker.get(new_batch.worker_id, {})
        parameter_changed = bool(change.get("parameter_changed")) or change.get("change") == "added"
        batch_size_changed = bool(change.get("batch_size_changed"))
        direct_worker_dependency = parameter_changed or batch_size_changed
        old_batch_id = old_membership.get((new_batch.worker_id, new_batch.arrival_seqs))
        same_batch = old_batch_id is not None and old_batch_id in old_batch_map
        if same_batch and not parameter_changed:
            report.reusable_batch_ids.append(new_batch.batch_id)
            continue

        dependency_level = "direct" if direct_worker_dependency else "indirect"
        causes = []
        if parameter_changed:
            causes.append("new_parameter_basis")
        if batch_size_changed:
            causes.append("new_batch_size_boundary")
        if not same_batch:
            causes.append("new_or_reshaped_membership")
        report.recompute_batch_ids.append(new_batch.batch_id)
        for request_id in new_batch.request_ids:
            if request_id not in report.recompute_request_ids:
                report.recompute_request_ids.append(request_id)
            old_loc = old_assignment.get(request_id)
            new_loc = new_assignment.get(request_id)
            if old_loc != new_loc and request_id not in report.reattribute_request_ids:
                report.reattribute_request_ids.append(request_id)
        report.batch_nodes.append(
            {
                "batch_id": new_batch.batch_id,
                "worker_id": new_batch.worker_id,
                "batch_index": new_batch.batch_index,
                "identity": "new_active_batch",
                "dependency_level": dependency_level,
                "causes": causes,
                "basis": {
                    "exact_old_batch_id": old_batch_id,
                    "parameter_changed": parameter_changed,
                    "worker_change": change or {"changed_fields": []},
                },
            }
        )

    report.recompute_batch_ids = _unique(report.recompute_batch_ids)
    report.recompute_request_ids = _unique(report.recompute_request_ids)
    report.reattribute_request_ids = _unique(report.reattribute_request_ids)
    report.reusable_batch_ids = _unique(report.reusable_batch_ids)
    identity_order = {"old_frozen_batch": 0, "new_active_batch": 1}
    report.batch_nodes.sort(
        key=lambda node: (
            node.get("worker_id", ""),
            int(node.get("batch_index", 0)),
            identity_order.get(node.get("identity"), 9),
            node.get("batch_id", ""),
        )
    )
    report.request_nodes.sort(
        key=lambda node: (
            node.get("worker_id", ""),
            node.get("batch_id", ""),
            int(node.get("rank", 0)),
            node.get("request_id", ""),
        )
    )
    return report


def _assignment_map(plan: Plan | None) -> dict[str, dict[str, Any]]:
    if plan is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for batch in plan.batches:
        for rank, request_id in enumerate(batch.request_ids):
            result[request_id] = {
                "batch_id": batch.batch_id,
                "worker_id": batch.worker_id,
                "rank": rank,
                "arrival_seq": batch.arrival_seqs[rank],
            }
    return result


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
