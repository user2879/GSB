"""Focused end-to-end scenarios used by CLI evidence artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, canonical_digest
from .cluster import NodeCluster
from .engine import recompute_invalid_subset_local
from .errors import DiagnosticError
from .invalidation import (
    changed_nodes_between_versions,
    derive_invalidation,
    hash_records,
    verify_byte_unchanged,
)
from .leases import LeaseManager
from .ledger import Ledger
from .spec import Spec


def run_evolution_demo(
    spec: Spec,
    *,
    ledger: Ledger,
    cluster: NodeCluster,
) -> dict[str, Any]:
    manager = LeaseManager()
    chosen = ["s01_a_ingest", "s02_b_ingest"]
    granted: list[dict[str, Any]] = []
    lease_rows: list[dict[str, Any]] = []
    shard_rows: list[dict[str, Any]] = []

    for shard_id in chosen:
        shard = spec.shards[shard_id]
        node_id = shard["node_id"]
        node = spec.nodes[node_id]
        lease = manager.grant(
            shard_id=shard_id,
            node_id=node_id,
            capacity_version=1,
            resource_units=int(shard["resource_units"]),
            lease_limit=int(node["lease_limit"]),
            ttl_ms=600_000,
        )
        with cluster.client(node_id) as client:
            client.lease_grant(
                {
                    "lease_id": lease["lease_id"],
                    "shard_id": shard_id,
                    "capacity_version": 1,
                    "granted_monotonic_ms": lease["granted_monotonic_ms"],
                    "expires_monotonic_ms": lease["expires_monotonic_ms"],
                    "ttl_ms": 600_000,
                }
            )
        granted.append(lease)
        lease_rows.append(lease)
        shard_rows.append(
            {
                "shard_id": shard_id,
                "status": "leased_for_evolution_demo",
                "lease_id": lease["lease_id"],
                "node_id": node_id,
                "capacity_version": 1,
                "resource_units": lease["resource_units"],
            }
        )

    grant_batch = ledger.commit_batch(
        "evolution_demo_leases_granted",
        {"lease_ids": [lease["lease_id"] for lease in granted]},
        shard_states=shard_rows,
        lease_states=lease_rows,
    )
    ownership_before = {
        lease["lease_id"]: {
            "lease_id": lease["lease_id"],
            "shard_id": lease["shard_id"],
            "node_id": lease["node_id"],
            "capacity_version": lease["capacity_version"],
        }
        for lease in granted
    }

    v2 = spec.capacity_versions[2]["node_calibers"]
    node_reports: list[dict[str, Any]] = []
    for node_id in sorted(spec.nodes):
        caliber = v2[node_id]
        with cluster.client(node_id) as client:
            report = client.capacity_advance(
                {
                    "to_version": 2,
                    "parent_version": 1,
                    "concurrency_quota": caliber["concurrency_quota"],
                    "lease_limit": caliber["lease_limit"],
                    "fixed_point_decimals": caliber["fixed_point_decimals"],
                }
            )
        node_reports.append(report)

    revoked = manager.revoke_for_capacity(2)
    expired_after_revoke = manager.reconcile_expired()
    manager.assert_single_holder()
    revoke_batch = ledger.commit_batch(
        "capacity_version_advanced_revoke_old_leases",
        {
            "from_version": 1,
            "to_version": 2,
            "revoked_lease_ids": [lease["lease_id"] for lease in revoked],
            "expired_after_revoke": expired_after_revoke,
            "double_recovery_prevented": True,
        },
        lease_states=revoked,
        shard_states=[
            {
                "shard_id": lease["shard_id"],
                "status": "capacity_generation_revoked",
                "lease_id": lease["lease_id"],
                "node_id": lease["node_id"],
                "capacity_version": 2,
                "resource_units": 0,
            }
            for lease in revoked
        ],
    )

    cross_generation_error: dict[str, Any] | None = None
    old_lease = granted[0]
    try:
        with cluster.client(old_lease["node_id"]) as client:
            client.lease_renew(
                {
                    "lease_id": old_lease["lease_id"],
                    "expected_version": 1,
                    "max_renewals": int(spec.raw["max_renewals"]),
                }
            )
    except DiagnosticError as exc:
        cross_generation_error = exc.to_dict()

    changed_nodes = changed_nodes_between_versions(spec, 1, 2)
    propagation = derive_invalidation(spec, changed_node_ids=changed_nodes)
    before_records = recompute_invalid_subset_local(spec, list(spec.shards), target_capacity_version=1)
    before_hashes = hash_records(before_records)
    after_records = dict(before_records)
    recomputed = recompute_invalid_subset_local(
        spec, propagation["invalid_shard_ids"], target_capacity_version=2
    )
    after_records.update(recomputed)
    after_hashes = hash_records(after_records)
    verification = verify_byte_unchanged(
        before_hashes, after_hashes, allowed_changed=propagation["invalid_shard_ids"]
    )

    current_leases = ledger.lease_states()
    ownership_after = {
        lease_id: {
            "lease_id": lease_id,
            "shard_id": lease["shard_id"],
            "node_id": lease["node_id"],
            "capacity_version": lease["capacity_version"],
        }
        for lease_id, lease in current_leases.items()
        if lease_id in ownership_before
    }
    ownership_unchanged = canonical_digest(ownership_before) == canonical_digest(ownership_after)

    return {
        "ok": (
            verification["ok"]
            and cross_generation_error is not None
            and cross_generation_error.get("code") == "E_LEASE"
            and manager.used_units() == 0
            and expired_after_revoke == []
            and ownership_unchanged
        ),
        "grant_batch": grant_batch,
        "revoke_batch": revoke_batch,
        "initial_used_units": sum(int(lease["resource_units"]) for lease in granted),
        "final_active_used_units": manager.used_units(),
        "revoked_leases": revoked,
        "expired_after_revoke": expired_after_revoke,
        "node_capacity_reports": node_reports,
        "cross_generation_renewal_error": cross_generation_error,
        "ownership_before_digest": canonical_digest(ownership_before),
        "ownership_after_digest": canonical_digest(ownership_after),
        "ownership_identity_unchanged": ownership_unchanged,
        "immutable_grant_batch_preserved": ledger.batches()[grant_batch["batch_seq"] - 1]["chain_digest"]
        == grant_batch["chain_digest"],
        "invalidation": propagation,
        "verification": verification,
    }


def run_and_write_demo(
    spec: Spec,
    *,
    db_path: str | Path,
    out_path: str | Path,
    log_dir: str | Path,
    fresh: bool = False,
) -> dict[str, Any]:
    db = Path(db_path)
    if fresh:
        for suffix in ("", "-wal", "-shm"):
            target = Path(str(db) + suffix)
            if target.exists():
                target.unlink()
    ledger = Ledger.open(db, expected_spec_digest=spec.spec_digest)
    cluster = NodeCluster(spec, log_dir=log_dir)
    try:
        cluster.start()
        result = run_evolution_demo(spec, ledger=ledger, cluster=cluster)
        ledger.export_snapshot(Path(out_path).with_suffix(".snapshot.json"))
        atomic_write_json(out_path, result)
        return result
    finally:
        cluster.stop()
        ledger.close()
