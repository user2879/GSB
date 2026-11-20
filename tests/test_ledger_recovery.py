import json

import pytest

from chaos import recover, timeline
from chaos.errors import E_SNAPSHOT, ChaosError
from chaos.ledger import Ledger
from chaos.structlog import StructLog


def make(tmp_path, spec_obj):
    log = StructLog(tmp_path / "run.jsonl")
    entries = timeline.generate(spec_obj)
    tdigest = timeline.timeline_digest(entries)
    return log, entries, tdigest


def test_fresh_ledger_initializes(tmp_path, spec_obj):
    log, entries, tdigest = make(tmp_path, spec_obj)
    ledger, report = recover.reconcile(spec_obj, tdigest, tmp_path / "l.db",
                                       tmp_path / "snap", log)
    assert not report.replaced
    meta = ledger.meta_dict()
    assert meta["seed"] == str(spec_obj.seed)
    assert meta["timeline_digest"] == tdigest
    assert ledger.verify_digest()
    ledger.close()


def test_uncommitted_active_marked_unknown(tmp_path, spec_obj):
    log, entries, tdigest = make(tmp_path, spec_obj)
    ledger, _ = recover.reconcile(spec_obj, tdigest, tmp_path / "l.db",
                                  tmp_path / "snap", log)
    e = entries[0]
    with ledger.transaction() as cur:
        cur.execute(
            "INSERT INTO injections(seq,stable_key,itype,target,params_json,"
            " planned_start_ms,state,committed,activated_mono_ms)"
            " VALUES(?,?,?,?,?,?, 'active',0,1)",
            (e.seq, e.stable_key, e.type, e.target,
             json.dumps(e.params, sort_keys=True), e.planned_start_ms))
        cur.execute(
            "INSERT INTO transitions(inj_seq,from_state,to_state,reason,"
            " mono_ms) VALUES(?,?,?,?,1)",
            (e.seq, "pending", "active", "activated"))
    ledger.close()
    ledger2, report = recover.reconcile(spec_obj, tdigest, tmp_path / "l.db",
                                        tmp_path / "snap", log)
    assert report.unknown == [e.seq]
    assert report.redo_plan == [e.seq]
    row = ledger2.conn.execute(
        "SELECT state, business_code FROM injections WHERE seq=?",
        (e.seq,)).fetchone()
    assert row[0] == "unknown"
    assert row[1] == "E_RECOVER_UNKNOWN"
    ledger2.close()


def test_tampered_ledger_raises_e_snapshot(tmp_path, spec_obj):
    log, entries, tdigest = make(tmp_path, spec_obj)
    ledger, _ = recover.reconcile(spec_obj, tdigest, tmp_path / "l.db",
                                  tmp_path / "snap", log)
    ledger.close()
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "l.db"))
    conn.execute(
        "INSERT INTO injections(seq,stable_key,itype,target,params_json,"
        " planned_start_ms,state,committed) VALUES(99,'x','latency','a',"
        " '{}',0,'failed',1)")
    conn.commit()
    conn.close()
    with pytest.raises(ChaosError) as exc:
        recover.reconcile(spec_obj, tdigest, tmp_path / "l.db",
                          tmp_path / "snap", log)
    assert exc.value.code == E_SNAPSHOT


def test_seed_drift_triggers_full_replace(tmp_path, spec_obj):
    log, entries, tdigest = make(tmp_path, spec_obj)
    ledger, _ = recover.reconcile(spec_obj, tdigest, tmp_path / "l.db",
                                  tmp_path / "snap", log)
    e = entries[0]
    with ledger.transaction() as cur:
        cur.execute(
            "INSERT INTO injections(seq,stable_key,itype,target,params_json,"
            " planned_start_ms,state,committed) VALUES(?,?,?,?,?,?,"
            " 'reverted',1)",
            (e.seq, e.stable_key, e.type, e.target,
             json.dumps(e.params, sort_keys=True), e.planned_start_ms))
    ledger.close()
    spec_obj.seed += 1
    entries2 = timeline.generate(spec_obj)
    tdigest2 = timeline.timeline_digest(entries2)
    ledger2, report = recover.reconcile(spec_obj, tdigest2, tmp_path / "l.db",
                                        tmp_path / "snap", log)
    assert report.replaced
    fields = {d["field"] for d in report.diffs}
    assert "seed" in fields
    count = ledger2.conn.execute("SELECT COUNT(*) FROM injections").fetchone()
    assert count[0] == 0  # full replace, not merge
    ledger2.close()
