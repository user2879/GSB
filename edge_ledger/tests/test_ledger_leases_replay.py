from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from edge_ledger.errors import LeaseError, ReplayError
from edge_ledger.ledger import Ledger
from edge_ledger.leases import LeaseService
from edge_ledger.replay import verify_ledger


def test_snapshot_digest_mismatch_is_replaced_with_readonly_backup(spec, tmp_path: Path):
    db = tmp_path / "ledger" / "edge.db"
    ledger = Ledger.open(db, content_digest=spec.content_digest)
    ledger.append_event("ping", quota_version=spec.quota_version)
    ledger.close()

    Ledger.open(db, content_digest="0" * 64).close()
    backups = list((tmp_path / "ledger" / "backups").glob("*.db"))
    assert len(backups) == 1
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o444

    replaced = Ledger.open(db, content_digest=spec.content_digest)
    meta = replaced.meta()
    assert meta["content_digest"] == spec.content_digest
    events = [row["event_type"] for row in replaced.fetch_all("events")]
    assert "snapshot_replaced" in events
    replaced.close()


def test_real_monotonic_expiry_releases_slot(spec, tmp_path: Path):
    ledger = Ledger.open(tmp_path / "edge.db", content_digest=spec.content_digest)
    service = LeaseService(spec, ledger)
    acq = service.acquire("node-c", "shard-0002", 2, spec.content.shard_size, ttl_ms=40)
    assert ledger.active_lease_rows()
    time.sleep(0.08)
    won = service.reap_expired()
    assert won
    row = ledger.conn.execute("SELECT state,decision_basis FROM leases WHERE lease_id=?", (acq.lease.lease_id,)).fetchone()
    assert row["state"] == "expired"
    assert row["decision_basis"] == "monotonic_now_ge_expires_at"
    assert not ledger.active_lease_rows()
    ledger.close()


def test_optimistic_renewal_version_conflict(spec, tmp_path: Path):
    ledger = Ledger.open(tmp_path / "edge.db", content_digest=spec.content_digest)
    service = LeaseService(spec, ledger)
    acq = service.acquire("node-b", "shard-0001", 1, spec.content.shard_size)
    renewed = service.commit_renewal_result(acq.lease, 1)
    assert renewed.lease_version == 2
    with pytest.raises(LeaseError) as exc:
        service.commit_renewal_result(acq.lease, 1)
    assert exc.value.code == "E_LEASE"
    ledger.close()


def test_replay_detects_tampered_digest(spec, tmp_path: Path):
    db = tmp_path / "edge.db"
    ledger = Ledger.open(db, content_digest=spec.content_digest)
    service = LeaseService(spec, ledger)
    service.acquire("node-a", "shard-0000", 0, spec.content.shard_size)
    ledger.conn.execute(
        """INSERT INTO shard_results(quota_version,node_id,shard_id,shard_seq,digest,
           content_bytes_digest,proof_json,source_format,status,unknown,event_seq)
           VALUES(1,'node-a','shard-0000',0,?,?,NULL,'v3','verified',0,
           (SELECT COALESCE(MAX(event_seq),1) FROM events))""",
        ("f" * 64, "f" * 64),
    )
    ledger.conn.commit()
    ledger.close()
    with pytest.raises(ReplayError) as exc:
        verify_ledger(spec.root, db)
    assert exc.value.code == "E_REPLAY"
    assert exc.value.shard_id == "shard-0000"
    assert exc.value.field in {"digest", "content_bytes_digest"}
