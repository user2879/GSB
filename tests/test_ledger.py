from __future__ import annotations

from faultisolation.clock import LogicalClock
from faultisolation.engine import Engine
from faultisolation.errors import LedgerError
from faultisolation.ledger import LEDGER_FORMAT_VERSION, Ledger
from faultisolation.replay import replay_path, verify_ledger
from faultisolation.recovery_demo import run_recovery_demo


def _write_full_ledger(bundle, path, batch_size=None):
    engine = Engine(bundle, LogicalClock(0))
    engine.run_event_path()
    ledger = Ledger(path, replace=True)
    size = batch_size or bundle.session.batch_size
    batch_id = 0
    for start in range(0, len(engine.outcomes), size):
        batch_id += 1
        chunk = engine.outcomes[start:start + size]
        lease_ids = {o.lease_id for o in chunk if o.lease_id}
        seqs = {o.seq for o in chunk}
        ledger.append_batch(
            batch_id=batch_id, outcomes=chunk,
            transitions=[t for t in engine.transitions if t.lease_id in lease_ids],
            waits=[w for w in engine.waits if w.seq in seqs],
            slot_snapshot={d: dict(a.slot_owner) for d, a in engine.arbiters.items() if a.slot_owner},
            clock_ms=max(o.clock_ms for o in chunk),
        )
    ledger.close()


def test_ledger_roundtrip_and_digest(small_bundle, tmp_path):
    path = tmp_path / "ledger.sqlite"
    _write_full_ledger(small_bundle, path)
    result = verify_ledger(small_bundle, path)
    assert result["digest_ok"] is True
    assert result["missing_seqs"] == []
    assert result["stored_events"] == len(small_bundle.events)


def test_tampered_digest_triggers_full_replace(small_bundle, tmp_path):
    path = tmp_path / "ledger.sqlite"
    _write_full_ledger(small_bundle, path)
    import sqlite3
    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE outcomes SET payload = payload || ' ' WHERE seq = 1")
    conn.commit()
    conn.close()
    ledger = Ledger(path)
    replaced = ledger.replace_if_digest_bad()
    assert replaced is True  # 摘要不一致：整体替换而非增量合并
    ledger.close()


def test_format_version_mismatch_replaces(small_bundle, tmp_path):
    path = tmp_path / "ledger.sqlite"
    _write_full_ledger(small_bundle, path)
    import sqlite3
    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE meta SET value='999' WHERE key='format_version'")
    conn.commit()
    conn.close()
    ledger = Ledger(path)
    assert ledger.replaced_on_open is True
    row = ledger.conn.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
    assert int(row[0]) == LEDGER_FORMAT_VERSION
    ledger.close()


def test_recovery_demo_forced_kill(tmp_path):
    from faultisolation.samples import generate_sample, SMALL_PLAN
    from faultisolation.spec import load_bundle
    paths = generate_sample(SMALL_PLAN, tmp_path, global_lease_cap=3, batch_size=16)
    ledger_path = tmp_path / "recovery.sqlite"
    result = run_recovery_demo(str(paths["spec"]), str(paths["session"]),
                               str(ledger_path), split_ratio=0.5)
    assert result["committed_seq_before_kill"] == result["forced_kill_after_event"]
    assert result["unknown_from_seq"] == result["forced_kill_after_event"] + 1
    assert result["digest_ok_at_recovery"] is True
    assert result["rerun_verify"]["digest_ok"] is True
    assert result["rerun_verify"]["missing_seqs"] == []
