import sqlite3

from faultdiag.ledger import Ledger


def test_batch_commit_and_digest(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db", {"domains_version": "v1"})
    with ledger.batch():
        ledger.add_transition({"event_seq": 1, "kind": "instance", "entity": "s-a1",
                               "from": "unknown", "to": "unhealthy",
                               "reason": "fault:crash", "mono_ms": 1})
        ledger.add_transition({"event_seq": 1, "kind": "domain", "entity": "dom-a",
                               "from": "normal", "to": "isolating",
                               "reason": "threshold", "mono_ms": 1})
        ledger.add_event_state(1, {"s-a1": "unhealthy"}, {"dom-a": "isolating"},
                               {"dom-a": {}}, "digest0")
        ledger.set_cursor(1, 0)
    assert ledger.cursor() == (1, 0)
    assert ledger.meta()["format_version"] == "1"
    assert ledger.digest()
    state = ledger.restore_state()
    assert state["health"] == {"s-a1": "unhealthy"}
    assert state["domain_state"] == {"dom-a": "isolating"}
    ledger.close()


def test_batch_rollback_leaves_no_trace(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db", {})
    try:
        with ledger.batch():
            ledger.add_transition({"event_seq": 1, "kind": "instance",
                                   "entity": "s-a1", "from": "unknown",
                                   "to": "unhealthy", "reason": "r", "mono_ms": 1})
            ledger.set_cursor(1, 0)
            raise RuntimeError("simulated kill")
    except RuntimeError:
        pass
    assert ledger.cursor() == (0, 0)
    assert ledger.restore_state()["health"] == {}
    ledger.close()


def test_recover_marks_running_unknown(tmp_path):
    ledger = Ledger(tmp_path / "ledger.db", {})
    ledger.conn.execute(
        "INSERT INTO commands(cmd_seq, event_seq, instance, domain, action,"
        " attempt, status, enqueue_mono_ms, start_mono_ms, end_mono_ms)"
        " VALUES(1, 1, 's-a1', 'dom-a', 'isolate', 1, 'running', 0, 0, 0)")
    ledger.conn.commit()
    unknown = ledger.recover()
    assert unknown == [{"id": 1, "instance": "s-a1", "cmd_seq": 1}]
    row = ledger.conn.execute("SELECT status, error FROM commands").fetchone()
    assert row[0] == "unknown" and "interrupted" in row[1]
    ledger.close()


def test_format_version_mismatch(tmp_path):
    path = tmp_path / "ledger.db"
    ledger = Ledger(path, {})
    ledger.close()
    conn = sqlite3.connect(path)
    conn.execute("UPDATE meta SET value='99' WHERE key='format_version'")
    conn.commit()
    conn.close()
    import pytest

    from faultdiag.errors import DiagError
    with pytest.raises(DiagError):
        Ledger(path, {})
