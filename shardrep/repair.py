"""Block repair decisions, majority source selection and partial-repair reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import sha256_object
from .models import BlockDecl, Spec
from .registry import ReplicaRecord, ReplicaRegistry, apply_repair_task


@dataclass(frozen=True)
class RepairTask:
    task_id: str
    block_id: str
    seq: int
    logical_id: str
    source_shard_id: str
    source_node_id: str
    target_shard_id: str
    target_node_id: str
    bytes: int
    expected_digest: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "block_id": self.block_id,
            "seq": self.seq,
            "logical_id": self.logical_id,
            "source_shard_id": self.source_shard_id,
            "source_node_id": self.source_node_id,
            "target_shard_id": self.target_shard_id,
            "target_node_id": self.target_node_id,
            "bytes": self.bytes,
            "expected_digest": self.expected_digest,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BlockDecision:
    block_id: str
    category: str
    missing_seqs: tuple[int, ...]
    attempted_replicas: tuple[dict[str, Any], ...]
    consensus: tuple[dict[str, Any], ...]
    repair_task_ids: tuple[str, ...]
    disabled_targets: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "category": self.category,
            "missing_seqs": list(self.missing_seqs),
            "attempted_replicas": list(self.attempted_replicas),
            "consensus": list(self.consensus),
            "repair_task_ids": list(self.repair_task_ids),
            "disabled_targets": list(self.disabled_targets),
        }


@dataclass(frozen=True)
class RepairPlan:
    tasks: tuple[RepairTask, ...]
    decisions: tuple[BlockDecision, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tasks": [task.as_dict() for task in self.tasks],
            "decisions": [decision.as_dict() for decision in self.decisions],
        }


def plan_repair(spec: Spec, registry: ReplicaRegistry) -> RepairPlan:
    tasks: list[RepairTask] = []
    decisions: list[BlockDecision] = []
    for block in spec.blocks:
        missing: list[int] = []
        attempted: list[dict[str, Any]] = []
        consensus: list[dict[str, Any]] = []
        block_task_ids: list[str] = []
        disabled_targets: list[str] = []
        for logical_id in spec.logicals_for_block(block):
            logical = spec.logical_shards[logical_id]
            records = list(registry.records_for_logical(logical_id))
            valid = [record for record in records if record.status == "ok"]
            votes: dict[str, int] = {}
            for record in records:
                if record.actual_digest is not None:
                    votes[record.actual_digest] = votes.get(record.actual_digest, 0) + 1
            consensus.append(
                {
                    "seq": logical.seq,
                    "logical_id": logical_id,
                    "expected_digest": logical.digest,
                    "usable_digest": logical.digest if valid else None,
                    "usable_count": len(valid),
                    "digest_votes": dict(sorted(votes.items())),
                }
            )
            if not valid:
                missing.append(logical.seq)
                for record in records:
                    attempted.append(
                        {
                            "block_id": block.block_id,
                            "seq": logical.seq,
                            "logical_id": logical_id,
                            "shard_id": record.shard_id,
                            "node_id": record.node_id,
                            "status": record.status,
                            "actual_digest": record.actual_digest,
                            "actual_bytes": record.actual_bytes,
                        }
                    )
                continue

            source = min(valid, key=lambda record: (record.node_id, record.shard_id))
            for target in records:
                if target.status == "ok":
                    continue
                target_node = spec.node_by_id[target.node_id]
                task_id = f"repair:{block.block_id}:{logical.seq}:{target.shard_id}"
                if not target_node.repair_enabled:
                    disabled_targets.append(target.shard_id)
                    continue
                task = RepairTask(
                    task_id=task_id,
                    block_id=block.block_id,
                    seq=logical.seq,
                    logical_id=logical_id,
                    source_shard_id=source.shard_id,
                    source_node_id=source.node_id,
                    target_shard_id=target.shard_id,
                    target_node_id=target.node_id,
                    bytes=logical.byte_count,
                    expected_digest=logical.digest,
                    reason=target.status,
                )
                tasks.append(task)
                block_task_ids.append(task.task_id)

        if missing:
            category = "unrecoverable"
        elif disabled_targets:
            category = "partial"
        else:
            category = "full"
        decisions.append(
            BlockDecision(
                block_id=block.block_id,
                category=category,
                missing_seqs=tuple(sorted(missing)),
                attempted_replicas=tuple(attempted),
                consensus=tuple(consensus),
                repair_task_ids=tuple(block_task_ids),
                disabled_targets=tuple(sorted(disabled_targets)),
            )
        )
    return RepairPlan(
        tasks=tuple(sorted(tasks, key=lambda task: (task.block_id, task.seq, task.target_shard_id))),
        decisions=tuple(decisions),
    )


def apply_and_report(
    spec: Spec,
    registry: ReplicaRegistry,
    *,
    logger: Any | None = None,
) -> dict[str, Any]:
    plan = plan_repair(spec, registry)
    results = [apply_repair_task(task, registry) for task in plan.tasks]
    result_by_task = {result["task_id"]: result for result in results}

    categories: dict[str, list[str]] = {"full": [], "partial": [], "unrecoverable": []}
    block_details: list[dict[str, Any]] = []
    for decision in plan.decisions:
        block = spec.block_by_id[decision.block_id]
        records = _block_records(spec, registry, block)
        failed_task = any(
            not result_by_task.get(task_id, {"ok": False})["ok"]
            for task_id in decision.repair_task_ids
        )
        if decision.missing_seqs:
            category = "unrecoverable"
        elif decision.disabled_targets or failed_task or any(record.status != "ok" for record in records):
            category = "partial"
        else:
            category = "full"
        categories[category].append(decision.block_id)
        status_counts: dict[str, int] = {}
        for record in records:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
        detail = decision.as_dict()
        detail.update(
            {
                "category": category,
                "replica_count": len(records),
                "replica_status_counts": dict(sorted(status_counts.items())),
                "repair_results": [result_by_task[task_id] for task_id in decision.repair_task_ids],
            }
        )
        block_details.append(detail)
        if category == "unrecoverable" and logger is not None:
            logger.emit(
                "E_UNRECOVERABLE",
                shard_id=(
                    spec.seq_to_logical[decision.missing_seqs[0]]
                    if decision.missing_seqs
                    else decision.block_id
                ),
                version=spec.group_version,
                block_id=decision.block_id,
                missing_seqs=list(decision.missing_seqs),
                attempted_replicas=list(decision.attempted_replicas),
            )
        elif logger is not None:
            logger.emit(
                "E_REPAIR_OK" if category == "full" else "E_REPAIR_PARTIAL",
                shard_id=block.block_id,
                version=spec.group_version,
                block_id=block.block_id,
                category=category,
                repair_tasks=list(decision.repair_task_ids),
            )

    counts = {key: len(value) for key, value in categories.items()}
    total_records = sum(len(_block_records(spec, registry, block)) for block in spec.blocks)
    report = {
        "code": "E_REPORT",
        "block_count": len(spec.blocks),
        "counts": counts,
        "categories": {key: sorted(value) for key, value in categories.items()},
        "repair_tasks": [task.as_dict() for task in plan.tasks],
        "repair_results": results,
        "decisions": block_details,
        "replica_total": total_records,
        "repair_task_count": len(plan.tasks),
        "repair_success_count": sum(1 for result in results if result["ok"]),
        "self_consistent": counts["full"] + counts["partial"] + counts["unrecoverable"] == len(spec.blocks),
    }
    return report


def _block_records(spec: Spec, registry: ReplicaRegistry, block: BlockDecl) -> list[ReplicaRecord]:
    records: list[ReplicaRecord] = []
    for logical_id in spec.logicals_for_block(block):
        records.extend(registry.records_for_logical(logical_id))
    return sorted(records, key=lambda record: (record.seq, record.node_id, record.shard_id))


def block_replica_digest(spec: Spec, registry: ReplicaRegistry) -> str:
    blocks: list[dict[str, Any]] = []
    for block in spec.blocks:
        seqs: list[dict[str, Any]] = []
        for logical_id in spec.logicals_for_block(block):
            records = registry.records_for_logical(logical_id)
            usable = sorted(record.actual_digest for record in records if record.status == "ok")
            seqs.append(
                {
                    "seq": spec.logical_shards[logical_id].seq,
                    "logical_id": logical_id,
                    "expected_digest": spec.logical_shards[logical_id].digest,
                    "usable_digests": usable,
                }
            )
        blocks.append({"block_id": block.block_id, "seqs": seqs})
    return sha256_object(blocks)


def repair_report_digest(report: dict[str, Any]) -> str:
    return sha256_object(report)
