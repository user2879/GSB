from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import pytest

from gsb.errors import DivergenceError, SnapshotError
from gsb.fixed import analyze_fixed_error, water_level
from gsb.ledger import (
    Ledger,
    compare_record_sets,
    inspect_snapshot,
    restore_snapshot,
)
from gsb.samples import canonical_json_bytes, sha256_hex


def make_record(
    shard_id: str,
    lease_id: str,
    *,
    session_version: int = 1,
    aggregate_digest: str = "a" * 64,
    progress_cursor: int = 1,
) -> dict:
    water = water_level("node-a", 3, 10, 1000).as_dict()
    error_model = analyze_fixed_error(2, 1000).as_dict()
    record = {
        "format_version": 1,
        "session_version": session_version,
        "shard_id": shard_id,
        "node_id": "node-a",
        "capacity_version": "cap-test",
        "aggregate_digest": aggregate_digest,
        "sample_digest": "b" * 64,
        "sample_row_count": 10,
        "participating_rows": 10,
        "fixed_scale": 1000,
        "session_fixed_scale": 1000,
        "group_count": 2,
        "resource_units": 3,
        "lease": {
            "lease_id": lease_id,
            "node_id": "node-a",
            "shard_id": shard_id,
            "units": 3,
            "status": "released",
            "reason": "completed",
        },
        "water": water,
        "error_attribution": {
            "segments_fp": error_model["components_fp"],
            "total_bound_fp": error_model["total_bound_fp"],
            "dominant_segment": error_model["dominant_segment"],
        },
        "progress_cursor": progress_cursor,
        "node_elapsed_ms": 1,
        "network_elapsed_ms": 2,
        "status": "committed",
    }
    record["content_digest"] = sha256_hex(canonical_json_bytes(record))
    return record


def _commit_one(ledger: Ledger, shard_id: str, lease_id: str) -> dict:
    now = 1_000_000
    ledger.reserve_lease(
        lease_id=lease_id,
        shard_id=shard_id,
        node_id="node-a",
        units=3,
        issued_ns=now,
        deadline_ns=now + 1_000_000_000,
    )
    record = make_record(shard_id, lease_id)
    ledger.complete_shard(record, lease_id=lease_id)
    return record


def test_atomic_complete_writes_all_fields_in_one_transaction(tmp_path) -> None:
    ledger = Ledger(tmp_path / "ledger.db", session_version=1)
    record = _commit_one(ledger, "shard-a", "lease-a")
    assert ledger.records()["shard-a"]["aggregate_digest"] == record["aggregate_digest"]
    row = ledger.conn.execute(
        "SELECT format_version,content_digest,aggregate_digest,lease_id,resource_units,"
        "water_json,error_json,progress_cursor FROM records WHERE shard_id='shard-a'"
    ).fetchone()
    assert row[0] == 1
    assert row[1] == record["content_digest"]
    assert row[3] == "lease-a"
    assert row[4] == 3
    assert row[7] == 1
    lease = ledger.conn.execute("SELECT status,close_reason FROM leases WHERE lease_id='lease-a'").fetchone()
    assert lease == ("released", "completed")
    assert ledger.conn.execute("SELECT COUNT(*) FROM uncommitted").fetchone()[0] == 0
    ledger.close()


def test_forced_termination_keeps_committed_and_reclaims_unknown(tmp_path) -> None:
    db = tmp_path / "crash.db"
    ledger = Ledger(db, session_version=1)
    _commit_one(ledger, "good-shard", "good-lease")
    ledger.close()

    env = dict(os.environ)
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gsb.crash_helper",
            "--ledger",
            str(db),
            "--lease-id",
            "crash-lease",
            "--shard-id",
            "crash-shard",
            "--units",
            "5",
        ],
        env=env,
        check=False,
    )
    assert result.returncode == 70

    reopened = Ledger(db, session_version=1)
    report = reopened.recover()
    assert report["committed_shards"] == ["good-shard"]
    assert report["unknown_shards"][0]["shard_id"] == "crash-shard"
    assert report["unknown_shards"][0]["requires_rerun"] is True
    assert report["reclaimed_units"] == 5
    assert report["exit_report_available"] is False
    assert reopened.conn.execute("SELECT COUNT(*) FROM uncommitted").fetchone()[0] == 0
    assert reopened.records()["good-shard"]["aggregate_digest"] == "a" * 64
    second_recovery = reopened.recover()
    assert second_recovery["reclaimed_units"] == 0
    assert second_recovery["unknown_shards"][0]["shard_id"] == "crash-shard"
    reopened.close()


def test_snapshot_restore_is_whole_replacement_not_merge(tmp_path) -> None:
    source = Ledger(tmp_path / "source.db", session_version=1)
    _commit_one(source, "source-shard", "source-lease")
    snapshot = tmp_path / "snapshot.db"
    source.create_snapshot(snapshot)
    source.close()
    assert inspect_snapshot(snapshot)["record_count"] == 1

    target = Ledger(tmp_path / "target.db", session_version=9)
    target.import_records([make_record("stale-shard", "stale-lease", session_version=9)])
    target.close()
    report = restore_snapshot(snapshot, tmp_path / "target.db", expected_version=1)
    assert report["whole_replacement"] is True
    assert report["merge_used"] is False
    assert report["version_mismatch"] is True
    assert report["digest_mismatch"] is True
    restored = Ledger(tmp_path / "target.db", session_version=1)
    records = restored.records()
    assert set(records) == {"source-shard"}
    assert records["source-shard"]["aggregate_digest"] == "a" * 64
    restored.close()

    with pytest.raises(SnapshotError):
        restore_snapshot(snapshot, tmp_path / "other.db", expected_version=2)


def test_snapshot_digest_tamper_is_e_snapshot(tmp_path) -> None:
    source = Ledger(tmp_path / "source.db", session_version=1)
    _commit_one(source, "source-shard", "source-lease")
    snapshot = tmp_path / "snapshot.db"
    source.create_snapshot(snapshot)
    source.close()
    tampered = tmp_path / "tampered.db"
    shutil.copy2(snapshot, tampered)
    conn = sqlite3.connect(tampered)
    conn.execute("UPDATE meta SET value='deadbeef' WHERE key='snapshot_digest'")
    conn.commit()
    conn.close()
    with pytest.raises(SnapshotError) as captured:
        inspect_snapshot(tampered)
    assert captured.value.code == "E_SNAPSHOT"


def test_compare_record_sets_reports_first_divergence_field() -> None:
    expected = {"a": make_record("a", "lease-a"), "b": make_record("b", "lease-b", progress_cursor=2)}
    actual = copy.deepcopy(expected)
    actual["a"]["water"]["utilization_fp"] += 1
    with pytest.raises(DivergenceError) as captured:
        compare_record_sets(actual, expected)
    assert captured.value.code == "E_DIVERGE"
    assert captured.value.first_shard == "a"
    assert captured.value.first_field == "water"
