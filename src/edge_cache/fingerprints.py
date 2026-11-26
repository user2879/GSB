"""Content-addressed cache keys and precise invalidation.

Every reusable artifact (node verification digest, authoritative receipt,
shard conclusion and per-version serialization) gets a key derived from:

* upstream content fingerprints (the shard and its whole dependency closure)
* the quota-version fingerprint (named version + chain position)
* the fixed-point width / quantile档位
* the exchange format version (c1/c2/c3)

Replacing one shard, changing one node's quota version or switching the
exchange format therefore invalidates only the keys that actually reference
the changed input; every other node reuses its persisted artifact.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from .config import BatchConfig, NodeConfig, ShardConfig
from .compute import shard_content_fingerprint
from .receipts import C1, C2, C3

EXCHANGE_VERSION = {C1: 1, C2: 2, C3: 3}


def _hash_blob(parts: Iterable[Any]) -> str:
    blob = json.dumps(list(parts), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def upstream_closure(config: BatchConfig, shard_id: str) -> tuple[str, ...]:
    """All transitive upstream shard ids, including the shard itself."""

    seen: set[str] = set()
    order: list[str] = []

    def walk(current: str) -> None:
        if current in seen:
            return
        seen.add(current)
        shard = config.shard(current)
        for parent in shard.upstream:
            walk(parent)
        order.append(current)

    walk(shard_id)
    return tuple(order)


def shard_chain_fingerprint(config: BatchConfig, shard_id: str) -> str:
    closure = upstream_closure(config, shard_id)
    parts = [shard_content_fingerprint(config.shard(item)) for item in closure]
    return _hash_blob(["shard-chain", closure, parts])


def quota_fingerprint(config: BatchConfig, quota_version: str) -> str:
    seq = config.quota_seq(quota_version)
    chain: list[str] = []
    current = quota_version
    by_name = {quota.version: quota for quota in config.quotas}
    while current is not None:
        chain.append(current)
        current = by_name[current].parent  # type: ignore[assignment]
    return _hash_blob(["quota", list(reversed(chain)), seq])


def node_profile_fingerprint(node: NodeConfig) -> str:
    return _hash_blob([
        "node-profile",
        node.node_id,
        node.fixed_point,
        node.quant_level,
        node.capacity_slots,
        node.lease_limit,
    ])


def node_summary_key(config: BatchConfig, node: NodeConfig, shard_id: str) -> str:
    return _hash_blob([
        "node-summary",
        node.node_id,
        shard_chain_fingerprint(config, shard_id),
        quota_fingerprint(config, node.quota_version),
        node_profile_fingerprint(node),
    ])


def authority_key(config: BatchConfig, shard_id: str) -> str:
    return _hash_blob([
        "authority",
        shard_chain_fingerprint(config, shard_id),
        sorted(
            quota_fingerprint(config, node.quota_version)
            for node in config.nodes
        ),
        sorted(node_profile_fingerprint(node) for node in config.nodes),
    ])


def conclusion_key(config: BatchConfig, shard_id: str) -> str:
    return _hash_blob([
        "conclusion",
        shard_chain_fingerprint(config, shard_id),
        authority_key(config, shard_id),
    ])


def serialization_key(config: BatchConfig, shard_id: str, fmt: str) -> str:
    return _hash_blob([
        "serialization",
        conclusion_key(config, shard_id),
        fmt,
        EXCHANGE_VERSION[fmt],
    ])


def ledger_entry_key(config: BatchConfig, shard_id: str) -> str:
    return _hash_blob([
        "ledger-entry",
        conclusion_key(config, shard_id),
        serialization_key(config, shard_id, C3),
    ])
