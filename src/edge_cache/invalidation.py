"""Precise invalidation of content-addressed artifacts.

Mutations (shard replacement, node quota/profile change, format switch)
invalidate only keys that reference the changed input.  The function returns
the set of affected shard/node pairs plus a human-readable reason; the engine
then deletes exactly those artifacts and records every invalidation in the
reuse audit stream.  Everything not listed is reused byte-for-byte.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import BatchConfig
from .fingerprints import (
    authority_key,
    conclusion_key,
    ledger_entry_key,
    node_summary_key,
    serialization_key,
    upstream_closure,
)
from .receipts import C1, C2, C3


@dataclass
class InvalidationPlan:
    artifact_keys: set[str]
    affected_shards: set[str]
    affected_nodes: set[str]
    reasons: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_keys": sorted(self.artifact_keys),
            "affected_shards": sorted(self.affected_shards),
            "affected_nodes": sorted(self.affected_nodes),
            "reasons": self.reasons,
        }


def _shard_downstream(config: BatchConfig, shard_id: str) -> set[str]:
    """Shards whose dependency closure contains ``shard_id`` (incl. itself)."""

    dependents: set[str] = set()
    for shard in config.shards:
        if shard_id in upstream_closure(config, shard.shard_id):
            dependents.add(shard.shard_id)
    return dependents


def invalidate_shard(config: BatchConfig, shard_id: str,
                     plan: InvalidationPlan) -> InvalidationPlan:
    affected = _shard_downstream(config, shard_id)
    plan.affected_shards.update(affected)
    for node in config.nodes:
        for current in affected:
            plan.artifact_keys.add(node_summary_key(config, node, current))
    for current in affected:
        plan.artifact_keys.add(authority_key(config, current))
        plan.artifact_keys.add(conclusion_key(config, current))
        plan.artifact_keys.add(ledger_entry_key(config, current))
        for fmt in (C1, C2, C3):
            plan.artifact_keys.add(serialization_key(config, current, fmt))
    plan.reasons.append({
        "cause": "replace_shard",
        "shard_id": shard_id,
        "downstream": sorted(affected),
    })
    return plan


def invalidate_node_quota(config: BatchConfig, node_id: str,
                          plan: InvalidationPlan) -> InvalidationPlan:
    node = config.node(node_id)
    plan.affected_nodes.add(node_id)
    for shard in config.shards:
        plan.artifact_keys.add(node_summary_key(config, node, shard.shard_id))
        plan.artifact_keys.add(authority_key(config, shard.shard_id))
        plan.artifact_keys.add(conclusion_key(config, shard.shard_id))
        plan.artifact_keys.add(ledger_entry_key(config, shard.shard_id))
        for fmt in (C1, C2, C3):
            plan.artifact_keys.add(serialization_key(config, shard.shard_id, fmt))
    plan.reasons.append({"cause": "set_quota_version", "node_id": node_id,
                         "quota_version": node.quota_version})
    return plan


def invalidate_node_profile(config: BatchConfig, node_id: str,
                            plan: InvalidationPlan) -> InvalidationPlan:
    node = config.node(node_id)
    plan.affected_nodes.add(node_id)
    for shard in config.shards:
        plan.artifact_keys.add(node_summary_key(config, node, shard.shard_id))
        plan.artifact_keys.add(authority_key(config, shard.shard_id))
        plan.artifact_keys.add(conclusion_key(config, shard.shard_id))
        plan.artifact_keys.add(ledger_entry_key(config, shard.shard_id))
        for fmt in (C1, C2, C3):
            plan.artifact_keys.add(serialization_key(config, shard.shard_id, fmt))
    plan.reasons.append({"cause": "node-profile-change", "node_id": node_id,
                         "fixed_point": node.fixed_point,
                         "quant_level": node.quant_level})
    return plan


def invalidate_format(config: BatchConfig, plan: InvalidationPlan) -> InvalidationPlan:
    # Format switches only touch serialization artifacts; summaries and
    # conclusions remain reusable.
    for shard in config.shards:
        for fmt in (C1, C2, C3):
            plan.artifact_keys.add(serialization_key(config, shard.shard_id, fmt))
        plan.artifact_keys.add(ledger_entry_key(config, shard.shard_id))
    plan.reasons.append({"cause": "format_roundtrip"})
    return plan


def empty_plan() -> InvalidationPlan:
    return InvalidationPlan(set(), set(), set(), [])


def apply_plan(ledger: Any, plan: InvalidationPlan) -> list[str]:
    removed: list[str] = []
    for key in sorted(plan.artifact_keys):
        path = ledger.artifact_path(key)
        import os

        if os.path.exists(path):
            os.remove(path)
            removed.append(key)
        ledger.record_reuse("invalidation", key, {"plan": plan.to_dict()})
    return removed
