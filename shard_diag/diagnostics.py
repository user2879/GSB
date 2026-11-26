from __future__ import annotations

from typing import Any

from .canonical import content_digest
from .determinism import compute_shard_summary
from .errors import ReplayError
from .ledger import Ledger, verify_replay_view
from .models import DelayBatch, Spec
from .topology import stable_topological_order
from .water import WaterAttribution, candidate_water


def latest_batch_for_node(spec: Spec, node_id: str) -> DelayBatch | None:
    batches = [batch for batch in spec.delay_batches if batch.node_id == node_id]
    if not batches:
        return None
    return sorted(batches, key=lambda batch: (batch.primary_revision, batch.batch_id))[-1]


def water_for_shard(
    spec: Spec,
    shard_id: str,
    *,
    lease_rearrangements: int = 0,
) -> WaterAttribution:
    shard = spec.shard(shard_id)
    node = spec.node(shard.node_id)
    batch = latest_batch_for_node(spec, node.node_id)
    if batch is None:
        delay_ms = 0
        jitter_ms = 0
        revision_lag = 0
    else:
        delay_ms = batch.delay_ms
        jitter_ms = batch.jitter_ms
        revision_lag = batch.revision_lag
    return candidate_water(
        node_id=node.node_id,
        shard_id=shard.shard_id,
        delay_ms=delay_ms,
        jitter_ms=jitter_ms,
        revision_lag=revision_lag,
        max_revision_lag=spec.max_revision_lag,
        delay_limit_ms=spec.dispatch_delay_limit_ms,
        resource_units=shard.resource_units,
        lease_limit=node.lease_limit,
        scale_power=spec.scale_power,
        lease_rearrangements=lease_rearrangements,
    )


def build_replay_view(
    spec: Spec,
    shard_id: str,
    *,
    lease_rearrangements: int = 0,
) -> dict[str, Any]:
    shard = spec.shard(shard_id)
    node = spec.node(shard.node_id)
    summary = compute_shard_summary(node, shard)
    water = water_for_shard(
        spec, shard_id, lease_rearrangements=lease_rearrangements
    ).to_dict()
    return {
        "node_id": node.node_id,
        "capacity_version": node.capacity_version,
        "resource_units": shard.resource_units,
        "summary": summary.to_dict(),
        "water": water,
    }


def verify_ledger_rebuild(spec: Spec, ledger: Ledger) -> dict[str, dict[str, Any]]:
    order = stable_topological_order(spec.shards)
    expected: dict[str, dict[str, Any]] = {}
    actual: dict[str, dict[str, Any]] = {}
    for shard_id in order:
        record = ledger.get_outcome(shard_id)
        if record is None:
            continue
        stored_digest = record.get("content_digest")
        unsigned_record = {
            key: value for key, value in record.items() if key != "content_digest"
        }
        if stored_digest != content_digest(unsigned_record):
            raise ReplayError(
                "ledger row content digest does not match payload",
                object=shard_id,
                field="content_digest",
            )
        rearrangements = int(
            ((record.get("water") or {}).get("lease_rearrangements") or 0)
        )
        expected[shard_id] = build_replay_view(
            spec, shard_id, lease_rearrangements=rearrangements
        )
        actual[shard_id] = {
            "node_id": record.get("node_id"),
            "capacity_version": record.get("capacity_version"),
            "resource_units": record.get("resource_units"),
            "summary": record.get("summary"),
            "water": record.get("water"),
        }
    verify_replay_view(expected, actual, [shard_id for shard_id in order if shard_id in actual])
    return expected
