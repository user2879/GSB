from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import spec_error
from .spec import (
    CheckpointSpec,
    EventSpec,
    ParameterVersion,
    ShardSpec,
    SpecBundle,
)


BatchKey = tuple[str, str]


@dataclass
class BatchPlan:
    event: EventSpec
    shard_id: str
    batch_id: str
    sample_ids: tuple[int, ...]
    parameter_version: int
    generation: int
    lease_id: str
    checkpoint_id: str | None = None
    committed_checkpoint: str | None = None
    invalidated_by_rollback: bool = False

    @property
    def key(self) -> BatchKey:
        return (self.shard_id, self.batch_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.event.seq,
            "shard_id": self.shard_id,
            "batch_id": self.batch_id,
            "sample_ids": list(self.sample_ids),
            "parameter_version": self.parameter_version,
            "generation": self.generation,
            "lease_id": self.lease_id,
            "checkpoint_id": self.checkpoint_id,
            "invalidated_by_rollback": self.invalidated_by_rollback,
        }


@dataclass
class CheckpointPlan:
    event: EventSpec
    spec: CheckpointSpec
    batch_keys: list[BatchKey]
    rolled_back: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.event.seq,
            "checkpoint_id": self.spec.checkpoint_id,
            "parameter_version": self.spec.parameter_version,
            "generation": self.spec.generation,
            "batch_keys": [list(key) for key in self.batch_keys],
            "rolled_back": self.rolled_back,
        }


@dataclass
class RollbackPlan:
    event: EventSpec
    checkpoint_id: str
    target_version: int
    target_generation: int
    revoked_versions: list[tuple[int, int]]
    revoked_checkpoints: list[str]
    affected_committed_batches: list[BatchKey]
    generation_before: int
    generation_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.event.seq,
            "checkpoint_id": self.checkpoint_id,
            "target_version": self.target_version,
            "target_generation": self.target_generation,
            "revoked_versions": [
                {"parameter_version": version, "generation": generation}
                for version, generation in self.revoked_versions
            ],
            "revoked_checkpoints": list(self.revoked_checkpoints),
            "affected_committed_batches": [
                {"shard_id": shard_id, "batch_id": batch_id}
                for shard_id, batch_id in self.affected_committed_batches
            ],
            "generation_before": self.generation_before,
            "generation_after": self.generation_after,
        }


@dataclass
class EventPlan:
    batches: list[BatchPlan]
    batch_by_key: dict[BatchKey, BatchPlan]
    checkpoints: dict[str, CheckpointPlan]
    checkpoint_order: list[str]
    rollbacks: list[RollbackPlan]
    advances: list[dict[str, Any]]
    lease_renewals: list[dict[str, Any]]
    initial_versions: dict[str, int]
    final_versions: dict[str, int]
    final_generation: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "batches": [batch.to_dict() for batch in self.batches],
            "checkpoints": [
                self.checkpoints[checkpoint_id].to_dict()
                for checkpoint_id in self.checkpoint_order
            ],
            "rollbacks": [rollback.to_dict() for rollback in self.rollbacks],
            "advances": list(self.advances),
            "lease_renewals": list(self.lease_renewals),
            "initial_versions": dict(sorted(self.initial_versions.items())),
            "final_versions": dict(sorted(self.final_versions.items())),
            "final_generation": self.final_generation,
        }


def _event_error(event: EventSpec, detail: str, *, item: str | None = None) -> None:
    raise spec_error(detail, item=item or f"event:{event.seq}", line=event.line)


def build_event_plan(bundle: SpecBundle) -> EventPlan:
    shard_by_id = bundle.shards.by_id()
    versions = bundle.layers.by_version()
    checkpoint_specs = bundle.checkpoints.by_id()
    initial_versions = {
        shard.shard_id: shard.parameter_version for shard in bundle.shards.shards
    }
    current_versions = dict(initial_versions)
    generation_counter = 0
    open_batch_keys: list[BatchKey] = []
    batches: list[BatchPlan] = []
    batch_by_key: dict[BatchKey, BatchPlan] = {}
    checkpoints: dict[str, CheckpointPlan] = {}
    checkpoint_order: list[str] = []
    rollbacks: list[RollbackPlan] = []
    advances: list[dict[str, Any]] = []
    renewals: list[dict[str, Any]] = []
    prior_commit_generation = 0

    def version_for(version_number: int, event: EventSpec) -> ParameterVersion:
        version = versions.get(version_number)
        if version is None:
            _event_error(event, f"missing parameter_version {version_number}")
        return version

    for event in bundle.events.events:
        if event.type == "batch":
            shard_id = event.data["shard_id"]
            if shard_id not in shard_by_id:
                _event_error(event, f"batch references unknown shard {shard_id!r}")
            batch_id = event.data["batch_id"]
            version_number = event.data.get(
                "parameter_version", current_versions[shard_id]
            )
            version = version_for(version_number, event)
            generation = max(version.generation, generation_counter)
            lease_id = f"lease-{event.seq}-{shard_id}-{batch_id}"
            batch = BatchPlan(
                event=event,
                shard_id=shard_id,
                batch_id=batch_id,
                sample_ids=tuple(sorted(event.data["sample_ids"])),
                parameter_version=version_number,
                generation=generation,
                lease_id=lease_id,
            )
            batches.append(batch)
            batch_by_key[batch.key] = batch
            open_batch_keys.append(batch.key)
        elif event.type == "checkpoint":
            checkpoint_id = event.data["checkpoint_id"]
            spec = checkpoint_specs[checkpoint_id]
            mismatched = {
                shard_id: version
                for shard_id, version in current_versions.items()
                if version != spec.parameter_version
            }
            if mismatched:
                _event_error(
                    event,
                    f"checkpoint {checkpoint_id!r} expects parameter_version "
                    f"{spec.parameter_version}, but shard versions differ: {mismatched}",
                    item=checkpoint_id,
                )
            requested_generation = event.data.get("generation")
            if (
                requested_generation is not None
                and requested_generation != spec.generation
            ):
                _event_error(
                    event,
                    f"checkpoint {checkpoint_id!r} generation does not match manifest",
                    item=checkpoint_id,
                )
            if spec.generation <= prior_commit_generation:
                _event_error(
                    event,
                    "checkpoint commits must follow manifest generation order",
                    item=checkpoint_id,
                )
            prior_commit_generation = spec.generation
            plan = CheckpointPlan(
                event=event,
                spec=spec,
                batch_keys=list(open_batch_keys),
            )
            checkpoints[checkpoint_id] = plan
            checkpoint_order.append(checkpoint_id)
            for key in open_batch_keys:
                batch_by_key[key].checkpoint_id = checkpoint_id
                batch_by_key[key].committed_checkpoint = checkpoint_id
            open_batch_keys.clear()
        elif event.type == "rollback":
            checkpoint_id = event.data["checkpoint_id"]
            target_plan = checkpoints.get(checkpoint_id)
            if target_plan is None:
                _event_error(
                    event,
                    f"rollback targets checkpoint {checkpoint_id!r} before commit",
                    item=checkpoint_id,
                )
            target_spec = target_plan.spec
            target_version = version_for(target_spec.parameter_version, event)
            active_generation_ceiling = max(
                versions[version_number].generation
                for version_number in current_versions.values()
            )
            revoked = sorted(
                (
                    (version.parameter_version, version.generation)
                    for version in bundle.layers.versions
                    if version.generation > target_spec.generation
                    and version.generation <= active_generation_ceiling
                ),
                key=lambda pair: (-pair[1], pair[0]),
            )
            revoked_checkpoints = sorted(
                (
                    plan.spec.checkpoint_id
                    for plan in checkpoints.values()
                    if plan.spec.generation > target_spec.generation
                ),
                key=lambda checkpoint_id: -checkpoints[checkpoint_id].spec.generation,
            )
            revoked_versions = {pair[0] for pair in revoked}
            affected = sorted(
                (
                    batch.key
                    for batch in batches
                    if batch.committed_checkpoint is not None
                    and batch.parameter_version in revoked_versions
                    and batch.event.seq < event.seq
                ),
                key=lambda key: (key[0], batch_by_key[key].event.seq, key[1]),
            )
            generation_before = generation_counter
            generation_counter = max(
                generation_counter, target_version.generation, prior_commit_generation
            ) + 1
            for checkpoint_name in revoked_checkpoints:
                checkpoints[checkpoint_name].rolled_back = True
            for batch in batches:
                if batch.event.seq < event.seq and batch.parameter_version in revoked_versions:
                    batch.invalidated_by_rollback = True
            rollback = RollbackPlan(
                event=event,
                checkpoint_id=checkpoint_id,
                target_version=target_spec.parameter_version,
                target_generation=target_version.generation,
                revoked_versions=revoked,
                revoked_checkpoints=revoked_checkpoints,
                affected_committed_batches=affected,
                generation_before=generation_before,
                generation_after=generation_counter,
            )
            rollbacks.append(rollback)
            for shard_id in current_versions:
                current_versions[shard_id] = target_spec.parameter_version
        elif event.type == "advance":
            version_number = event.data["parameter_version"]
            version = version_for(version_number, event)
            shard_ids = event.data.get("shard_ids", sorted(current_versions))
            for shard_id in shard_ids:
                current_generation = versions[current_versions[shard_id]].generation
                if version.generation < current_generation:
                    _event_error(
                        event,
                        f"advance for shard {shard_id!r} moves backwards; use rollback",
                        item=f"event:{event.seq}",
                    )
                current_versions[shard_id] = version_number
            generation_counter = max(generation_counter, version.generation)
            advances.append(
                {
                    "seq": event.seq,
                    "parameter_version": version_number,
                    "generation": version.generation,
                    "shard_ids": list(shard_ids),
                }
            )
        elif event.type == "renew":
            key = (event.data["shard_id"], event.data["batch_id"])
            batch = batch_by_key[key]
            renewals.append(
                {
                    "seq": event.seq,
                    "shard_id": key[0],
                    "batch_id": key[1],
                    "lease_id": batch.lease_id,
                    "lease_version": event.data["lease_version"],
                }
            )
    return EventPlan(
        batches=batches,
        batch_by_key=batch_by_key,
        checkpoints=checkpoints,
        checkpoint_order=checkpoint_order,
        rollbacks=rollbacks,
        advances=advances,
        lease_renewals=renewals,
        initial_versions=initial_versions,
        final_versions=dict(current_versions),
        final_generation=generation_counter,
    )
