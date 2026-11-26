from __future__ import annotations

import os
import stat

from ckptsim.ledger import Ledger, SnapshotIdentity
from ckptsim.plan import build_event_plan
from ckptsim.spec import load_bundle


def _bundle(files, layers=None):
    return load_bundle(
        files.shards,
        layers or files.layers,
        files.checkpoints,
        files.events,
    )


def test_checkpoint_ownership_and_rollback_are_deterministic(spec_files) -> None:
    bundle = _bundle(spec_files)
    plan = build_event_plan(bundle)
    assert plan.checkpoints["cp-base"].batch_keys == [
        ("shard-a", "b-a1"),
        ("shard-b", "b-b1"),
    ]
    assert plan.checkpoints["cp-v2"].batch_keys == [
        ("shard-a", "b-a2"),
        ("shard-b", "b-b2"),
    ]
    rollback = plan.rollbacks[0]
    assert rollback.revoked_versions == [(2, 2)]
    assert rollback.revoked_checkpoints == ["cp-v2"]
    assert rollback.affected_committed_batches == [
        ("shard-a", "b-a2"),
        ("shard-b", "b-b2"),
    ]
    assert rollback.generation_after == 3
    assert plan.batch_by_key[("shard-a", "b-a3")].parameter_version == 1
    assert plan.batch_by_key[("shard-a", "b-a3")].generation == 3


def test_inflight_is_recovered_as_unknown_and_lease_reclaimed(spec_files, tmp_path) -> None:
    bundle = _bundle(spec_files)
    plan = build_event_plan(bundle)
    ledger_path = tmp_path / "ledger.sqlite"
    ledger = Ledger.open(
        ledger_path,
        identity=SnapshotIdentity.from_bundle(bundle),
        initial_versions=plan.initial_versions,
        initial_generation=0,
    )
    batch = plan.batches[0]
    ledger.begin_batch(
        batch,
        lease_id=batch.lease_id,
        lease_version=1,
        granted_ms=1,
        expires_ms=10_000,
    )
    ledger.close()
    reopened = Ledger.open(
        ledger_path,
        identity=SnapshotIdentity.from_bundle(bundle),
        initial_versions=plan.initial_versions,
        initial_generation=0,
    )
    try:
        recovered = reopened.recover_inflight("simulated hard kill")
        assert recovered == [
            {
                "shard_id": "shard-a",
                "batch_id": "b-a1",
                "lease_id": "lease-1-shard-a-b-a1",
                "status": "unknown",
            }
        ]
        record = reopened.get_batch("shard-a", "b-a1")
        assert record is not None and record["status"] == "unknown"
        attempts = reopened.list_attempts()
        assert attempts[0]["status"] == "unknown"
        lease_row = reopened.connection.execute(
            "SELECT status FROM leases WHERE lease_id=?", ("lease-1-shard-a-b-a1",)
        ).fetchone()
        assert lease_row["status"] == "released_recovered"
    finally:
        reopened.close()


def test_snapshot_identity_mismatch_replaces_and_backs_up(spec_files, tmp_path) -> None:
    bundle = _bundle(spec_files)
    plan = build_event_plan(bundle)
    ledger_path = tmp_path / "ledger.sqlite"
    ledger = Ledger.open(
        ledger_path,
        identity=SnapshotIdentity.from_bundle(bundle),
        initial_versions=plan.initial_versions,
        initial_generation=0,
    )
    ledger.close()
    changed = _bundle(spec_files, layers=spec_files.edited_layers)
    replacement = Ledger.open(
        ledger_path,
        identity=SnapshotIdentity.from_bundle(changed),
        initial_versions=plan.initial_versions,
        initial_generation=0,
    )
    try:
        report = replacement.snapshot_report
        assert report is not None and report.rebuilt
        fields = {item["field"] for item in report.differences}
        assert "parameter_version_dialect" in fields
        assert "content_digest" in fields
        assert report.backup_path is not None
        mode = stat.S_IMODE(os.stat(report.backup_path).st_mode)
        assert mode == 0o444
        assert replacement.list_batches() == []
    finally:
        replacement.close()

