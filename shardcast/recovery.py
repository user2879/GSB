"""Crash recovery and ledger replay based only on committed records."""

from __future__ import annotations

from dataclasses import dataclass

from .engine import Engine
from .errors import DivergeError
from .ledger import Ledger
from .propagation import compare_field_by_field
from .sample import digest_payload
from .spec import Spec


@dataclass
class RecoveryReport:
    committed: list[str]
    unknown: list[str]
    reclaimed_units: int
    replayed: list[str]
    mismatches: list[dict]

    def to_dict(self) -> dict:
        return {
            "committed": self.committed,
            "unknown": self.unknown,
            "reclaimed_units": self.reclaimed_units,
            "replayed": self.replayed,
            "mismatches": self.mismatches,
        }


def recover_and_replay(
    spec: Spec,
    ledger_path: str,
    *,
    killed_node: str | None = None,
    batch_size: int = 64,
) -> RecoveryReport:
    """Open the durable ledger, reclaim unknown work, then replay to parity.

    A worker that was force-killed cannot produce an exit report: only fully
    committed ledger rows are trusted; anything absent is marked unknown and
    reclaimed before rerun.
    """
    ledger = Ledger(ledger_path)
    created, replacement = ledger.initialize_snapshot(
        spec_digest=spec.structural_digest(),
        node_set_version=spec.node_set_version(),
        capacity_version=min(spec.global_budget),
    )
    if created and replacement is not None:
        raise DivergeError(
            "ledger snapshot belonged to another declaration and was replaced; "
            "recovery against a different declaration is refused",
            path="meta.spec_digest",
        )
    prior = ledger.all_outcomes()
    committed = sorted(spec_id for spec_id, row in prior.items() if row["status"] == "completed")
    known_ids = set(prior)
    unknown = sorted(shard.shard_id for shard in spec.shards if shard.shard_id not in known_ids)
    reclaimed_units = sum(spec.shard(shard_id).resource_units for shard_id in unknown)

    engine = Engine(spec, ledger, batch_size=batch_size)
    engine.start()
    try:
        if killed_node is not None:
            engine.pool.restart_node(killed_node)
        stored_version = ledger.current_capacity_version()
        if stored_version != engine.capacity.current.version:
            engine._replace_capacity(stored_version)
        _preseed_completed(engine, prior, committed)
        engine.arrived = {shard.shard_id for shard in spec.shards}
        engine._drain_until_settled()
        result = engine._build_result()
    finally:
        engine.stop()

    mismatches = _compare_with_ledger(spec, ledger, prior, result)
    replayed = sorted(
        shard_id
        for shard_id in result.outcomes
        if shard_id not in committed
    )
    return RecoveryReport(
        committed=committed,
        unknown=unknown,
        reclaimed_units=reclaimed_units,
        replayed=replayed,
        mismatches=mismatches,
    )


def _preseed_completed(
    engine: Engine, prior: dict[str, dict], committed: list[str]
) -> None:
    for shard_id in committed:
        row = prior[shard_id]
        runtime = engine.runtimes[shard_id]
        runtime.status = row["status"]
        runtime.node_id = row["node_id"]
        runtime.capacity_version = row["capacity_version"]
        runtime.lease_id = row["lease_id"]
        runtime.summary = row["summary"]
        runtime.stale = row["capacity_version"] < engine.capacity.current.version


def _rebuild_committed_leases(engine: Engine, prior: dict[str, dict]) -> None:
    """Only completed, non-revoked records may seed fresh worker lease state."""
    for shard_id in sorted(prior):
        row = prior[shard_id]
        if row["status"] != "completed" or row.get("lease_id") is None:
            continue
        lease = engine.lease_manager.grant(
            row["lease_id"],
            shard_id,
            row["node_id"],
            row["capacity_version"],
        )
        lease.lease_version = int(row.get("lease_version") or 1)
        engine.pool.install_lease(row["node_id"], lease.to_dict())
        engine.lease_manager.release(lease.lease_id, reason="recovered_committed")


def _compare_with_ledger(
    spec: Spec,
    ledger: Ledger,
    prior: dict[str, dict],
    result,
) -> list[dict]:
    current = ledger.all_outcomes()
    mismatches: list[dict] = []
    for shard_id in spec.shard_ids:
        before = prior.get(shard_id)
        after = current.get(shard_id)
        if after is None:
            mismatches.append({"shard_id": shard_id, "field": "row", "detail": "missing after replay"})
            raise DivergeError("ledger row missing after replay", path=f"outcomes.{shard_id}")
        comparable = ("node_id", "status", "capacity_version", "resource_units")
        for field_name in comparable:
            if before is not None and before[field_name] != after[field_name]:
                mismatches.append(
                    {
                        "shard_id": shard_id,
                        "field": field_name,
                        "before": before[field_name],
                        "after": after[field_name],
                    }
                )
                raise DivergeError(
                    f"replay divergence on {field_name}",
                    path=f"outcomes.{shard_id}.{field_name}",
                )
        if after["summary"] is not None:
            digest = digest_payload(after["summary"])
            if digest != after["summary_digest"]:
                raise DivergeError(
                    "summary digest mismatch after replay",
                    path=f"outcomes.{shard_id}.summary_digest",
                )
            runtime_summary = result.outcomes[shard_id]
            if runtime_summary["status"] != after["status"]:
                raise DivergeError(
                    "engine status differs from ledger",
                    path=f"outcomes.{shard_id}.status",
                )
    return mismatches


def compare_full_state(
    left: dict, right: dict, *, label_left: str, label_right: str
) -> None:
    """Strict field-by-field state comparison used by acceptance checks."""
    compare_field_by_field(
        {key: left[key] for key in sorted(left) if key in right},
        {key: right[key] for key in sorted(right) if key in left},
        context=f"{label_left}:{label_right}",
    )
