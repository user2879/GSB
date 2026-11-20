from __future__ import annotations

import json
import sqlite3

import pytest

from trace_diag.errors import ReplayError, SnapshotError
from trace_diag.ledger import Ledger
from trace_diag.replay import verify_ledger_run
from trace_diag.workflow import analyze_and_persist


def test_replay_reproduces_quantiles_attribution_and_snapshot(tmp_path, service_doc, policy_doc):
    db = tmp_path / "ledger.sqlite"
    result = analyze_and_persist(service_doc, policy_doc, db)
    verification = verify_ledger_run(db)
    assert verification["ok"] is True
    assert verification["run_id"] == result.run_id
    assert verification["services_checked"] == 2
    assert verification["snapshot_digest"] == result.view["run"]["snapshot_digest"]


def test_replay_locates_first_tampered_service_field(tmp_path, service_doc, policy_doc):
    db = tmp_path / "ledger.sqlite"
    result = analyze_and_persist(service_doc, policy_doc, db)
    connection = sqlite3.connect(db)
    row = connection.execute(
        "SELECT service_id, policy_id, bucket_index, payload_json FROM bucket_stats "
        "WHERE run_id=? ORDER BY service_id, policy_id, bucket_index LIMIT 1",
        (result.run_id,),
    ).fetchone()
    payload = json.loads(row[3])
    payload["sample_hits"] += 1
    connection.execute(
        "UPDATE bucket_stats SET payload_json=? "
        "WHERE run_id=? AND service_id=? AND policy_id=? AND bucket_index=?",
        (json.dumps(payload), result.run_id, row[0], row[1], row[2]),
    )
    connection.commit()
    connection.close()
    with pytest.raises(ReplayError) as captured:
        verify_ledger_run(db)
    assert captured.value.service_id == row[0]
    assert ".buckets" in captured.value.field
    assert ".sample_hits" in captured.value.field


def test_replay_surfaces_snapshot_digest_only_tamper(tmp_path, service_doc, policy_doc):
    db = tmp_path / "ledger.sqlite"
    result = analyze_and_persist(service_doc, policy_doc, db)
    with Ledger(db, mark_pending_unknown=False) as ledger:
        ledger.corrupt_snapshot_digest(result.run_id)
    with pytest.raises(SnapshotError) as captured:
        verify_ledger_run(db)
    assert captured.value.code == "E_SNAPSHOT"
