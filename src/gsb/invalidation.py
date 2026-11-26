from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from .canonical import digest_json
from .planner import CandidateSet
from .schema import BlockGroupDecl, Workload


@dataclass(frozen=True)
class PlanDependency:
    plan_id: str
    block_group_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"plan_id": self.plan_id, "block_group_ids": list(self.block_group_ids)}


@dataclass(frozen=True)
class BatchDependency:
    batch_id: str
    plan_ids: tuple[str, ...]
    block_group_ids: tuple[str, ...]
    receipt_digest: str
    output_digest: str
    state: str = "valid"

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "plan_ids": list(self.plan_ids),
            "block_group_ids": list(self.block_group_ids),
            "receipt_digest": self.receipt_digest,
            "output_digest": self.output_digest,
            "state": self.state,
        }


@dataclass(frozen=True)
class InvalidationEvent:
    sequence: int
    plan_version_before: int
    plan_version_after: int
    changes: tuple[dict[str, Any], ...]
    affected_block_groups: tuple[str, ...]
    invalidated_plan_ids: tuple[str, ...]
    invalidated_batch_ids: tuple[str, ...]
    preserved_batches: tuple[BatchDependency, ...]
    steps: tuple[dict[str, Any], ...]

    @property
    def digest(self) -> str:
        return digest_json(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "plan_version_before": self.plan_version_before,
            "plan_version_after": self.plan_version_after,
            "changes": list(self.changes),
            "affected_block_groups": list(self.affected_block_groups),
            "invalidated_plan_ids": list(self.invalidated_plan_ids),
            "invalidated_batch_ids": list(self.invalidated_batch_ids),
            "preserved_batches": [batch.as_dict() for batch in self.preserved_batches],
            "steps": list(self.steps),
        }


def histogram_fingerprint(group: BlockGroupDecl) -> str:
    return digest_json(
        {
            group.id: {
                column.name: column.histogram.as_dict()
                for column in group.columns
            }
        }
    )


def cost_weight_fingerprint(workload: Workload) -> str:
    model = workload.cost_model
    return digest_json(
        {
            "access_row": model.access_row_weight,
            "comparison": model.comparison_weight,
            "memory_byte": model.memory_byte_weight,
            "intermediate_row": model.intermediate_row_weight,
        }
    )


def scale_fingerprint(workload: Workload) -> str:
    model = workload.cost_model
    return digest_json({"fixed": model.fixed_scale, "batch_rows": model.batch_rows})


class InvalidationEngine:
    def __init__(
        self,
        workload: Workload,
        candidates: CandidateSet,
        *,
        data_fingerprints: dict[str, str] | None = None,
        histogram_fingerprints: dict[str, str] | None = None,
        plan_version: int = 0,
    ) -> None:
        self._workload = workload
        self._plan_version = plan_version
        self._sequence = 0
        self._data_fingerprints = {
            group.id: (
                data_fingerprints[group.id]
                if data_fingerprints is not None
                else group.fingerprint
            )
            for group in workload.query_block_groups()
        }
        self._histogram_fingerprints = {
            group.id: (
                histogram_fingerprints[group.id]
                if histogram_fingerprints is not None
                else histogram_fingerprint(group)
            )
            for group in workload.query_block_groups()
        }
        self._weight_fingerprint = cost_weight_fingerprint(workload)
        self._scale_fingerprint = scale_fingerprint(workload)
        self._plans = {
            plan.plan_id: PlanDependency(plan.plan_id, tuple(sorted(plan.tables)))
            for plan in candidates.plans
        }
        self._batches: dict[str, BatchDependency] = {}
        self._events: list[InvalidationEvent] = []

    @property
    def plan_version(self) -> int:
        return self._plan_version

    @property
    def events(self) -> tuple[InvalidationEvent, ...]:
        return tuple(self._events)

    def register_batch(
        self,
        *,
        batch_id: str,
        plan_ids: Iterable[str],
        block_group_ids: Iterable[str],
        receipt_digest: str,
        output_digest: str,
    ) -> BatchDependency:
        batch = BatchDependency(
            batch_id=batch_id,
            plan_ids=tuple(sorted(plan_ids)),
            block_group_ids=tuple(sorted(block_group_ids)),
            receipt_digest=receipt_digest,
            output_digest=output_digest,
            state="valid",
        )
        self._batches[batch_id] = batch
        return batch

    def update_batch(
        self,
        batch_id: str,
        *,
        receipt_digest: str,
        output_digest: str,
    ) -> BatchDependency:
        current = self._batches[batch_id]
        updated = replace(
            current,
            receipt_digest=receipt_digest,
            output_digest=output_digest,
            state="valid",
        )
        self._batches[batch_id] = updated
        return updated

    def batches(self) -> tuple[BatchDependency, ...]:
        return tuple(sorted(self._batches.values(), key=lambda batch: batch.batch_id))

    def apply_changes(self, changes: Iterable[dict[str, Any]]) -> InvalidationEvent:
        change_list = tuple(
            sorted(
                changes,
                key=lambda item: (str(item.get("kind")), str(item.get("block_group_id", ""))),
            )
        )
        effective: list[dict[str, Any]] = []
        affected_block_groups: set[str] = set()
        invalidate_all_plans = False
        for change in change_list:
            kind = change.get("kind")
            if kind in ("block_group_data", "block_group_histogram"):
                block_group_id = change.get("block_group_id")
                fingerprint = change.get("fingerprint")
                source = self._data_fingerprints if kind == "block_group_data" else self._histogram_fingerprints
                if not isinstance(block_group_id, str) or block_group_id not in source:
                    raise ValueError(f"unknown block group for {kind}: {block_group_id}")
                if not isinstance(fingerprint, str) or len(fingerprint) < 16:
                    raise ValueError("fingerprint must be a stable digest string")
                if source[block_group_id] == fingerprint:
                    continue
                old_fingerprint = source[block_group_id]
                source[block_group_id] = fingerprint
                affected_block_groups.add(block_group_id)
                effective.append(
                    {
                        "kind": kind,
                        "block_group_id": block_group_id,
                        "old_fingerprint": old_fingerprint,
                        "new_fingerprint": fingerprint,
                    }
                )
            elif kind == "cost_weights":
                fingerprint = change.get("fingerprint")
                if not isinstance(fingerprint, str) or len(fingerprint) < 16:
                    raise ValueError("fingerprint must be a stable digest string")
                if fingerprint == self._weight_fingerprint:
                    continue
                old = self._weight_fingerprint
                self._weight_fingerprint = fingerprint
                invalidate_all_plans = True
                effective.append({"kind": kind, "old_fingerprint": old, "new_fingerprint": fingerprint})
            elif kind == "scale_factors":
                fingerprint = change.get("fingerprint")
                if not isinstance(fingerprint, str) or len(fingerprint) < 16:
                    raise ValueError("fingerprint must be a stable digest string")
                if fingerprint == self._scale_fingerprint:
                    continue
                old = self._scale_fingerprint
                self._scale_fingerprint = fingerprint
                invalidate_all_plans = True
                effective.append({"kind": kind, "old_fingerprint": old, "new_fingerprint": fingerprint})
            else:
                raise ValueError(f"unsupported invalidation change kind: {kind}")

        before = self._plan_version
        if effective:
            self._plan_version += 1
            self._sequence += 1

        affected_plans: set[str] = set()
        if effective:
            if invalidate_all_plans:
                affected_plans.update(self._plans)
            for plan in self._plans.values():
                if set(plan.block_group_ids) & affected_block_groups:
                    affected_plans.add(plan.plan_id)

        invalidated_batches: list[BatchDependency] = []
        preserved_batches: list[BatchDependency] = []
        for batch in sorted(self._batches.values(), key=lambda item: (item.block_group_ids, item.plan_ids, item.batch_id)):
            invalid = False
            if effective:
                if invalidate_all_plans:
                    invalid = True
                elif set(batch.plan_ids) & affected_plans and set(batch.block_group_ids) & affected_block_groups:
                    invalid = True
            if invalid:
                updated = replace(batch, state="invalid")
                self._batches[batch.batch_id] = updated
                invalidated_batches.append(updated)
            else:
                preserved_batches.append(batch)

        steps: list[dict[str, Any]] = []
        for plan in sorted(affected_plans):
            triggers = []
            if invalidate_all_plans:
                triggers = sorted(
                    str(change["kind"]) for change in effective if change["kind"] in ("cost_weights", "scale_factors")
                )
                basis = "cost/scale fingerprint changed"
            else:
                triggers = sorted(affected_block_groups)
                basis = "block-group fingerprint changed"
            steps.append(
                {
                    "from": triggers if triggers else ["query"],
                    "to": f"plan:{plan}",
                    "basis": basis,
                }
            )
        for batch in invalidated_batches:
            steps.append(
                {
                    "from": [f"plan:{plan_id}" for plan_id in batch.plan_ids],
                    "to": f"batch:{batch.batch_id}",
                    "basis": (
                        "cost/scale changed for plan"
                        if invalidate_all_plans
                        else f"batch depends on block groups {list(affected_block_groups)}"
                    ),
                }
            )

        event = InvalidationEvent(
            sequence=self._sequence,
            plan_version_before=before,
            plan_version_after=self._plan_version,
            changes=tuple(effective),
            affected_block_groups=tuple(sorted(affected_block_groups)),
            invalidated_plan_ids=tuple(sorted(affected_plans)),
            invalidated_batch_ids=tuple(sorted(batch.batch_id for batch in invalidated_batches)),
            preserved_batches=tuple(preserved_batches),
            steps=tuple(steps),
        )
        self._events.append(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        return {
            "plan_version": self._plan_version,
            "sequence": self._sequence,
            "data_fingerprints": dict(sorted(self._data_fingerprints.items())),
            "histogram_fingerprints": dict(sorted(self._histogram_fingerprints.items())),
            "weight_fingerprint": self._weight_fingerprint,
            "scale_fingerprint": self._scale_fingerprint,
            "plans": [
                self._plans[plan_id].as_dict()
                for plan_id in sorted(self._plans)
            ],
            "batches": [batch.as_dict() for batch in self.batches()],
            "events": [event.as_dict() for event in self._events],
        }
