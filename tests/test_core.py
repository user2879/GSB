import sqlite3
import threading
import time

import pytest

from sharedbuf import buffer as buf
from sharedbuf.arbiter import Request
from sharedbuf.errors import (E_BUSY, E_FIELD, E_OWNER, E_RECOVER, E_REPLAY,
                              E_SCOPE, E_SEQ, E_TIMEOUT, E_VER, PlatformError)
from sharedbuf.lease import LeaseManager
from sharedbuf.platform import Platform
from sharedbuf.protocol import validate_message


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_cfg(**overrides):
    sectors = ["b0#0", "b0#1", "b1#0", "b1#1"]
    cfg = {
        "seed": 7,
        "sectors": sectors,
        "sector_index": {s: i for i, s in enumerate(sectors)},
        "owners": {"b0#0": "n0", "b1#0": "n0", "b0#1": "n1", "b1#1": "n1"},
        "lease_ttl_ms": 5000,
        "global_limit": 4,
        "node_quota": 2,
        "adjudication_timeout_s": 2.0,
        "adjudication_retries": 1,
    }
    cfg.update(overrides)
    return cfg


def make_platform(tmp_path, name="t.db", clock=None, node_runner=None, **kw):
    return Platform(make_cfg(**kw), db_path=str(tmp_path / name),
                    clock=clock or time.monotonic, node_runner=node_runner)


def op_msg(wl, **kw):
    msg = {"type": "op", "wl": wl, "lease_id": "L1", "node_id": "n0",
           "op": "read", "agg": "sum", "sectors": ["b0#0"],
           "seq_lo": 1, "seq_hi": 2, "delta": 0.0}
    msg.update(kw)
    return msg


def test_field_whitelist():
    wl = "wl1.1"
    validate_message(op_msg(wl), wl)

    bad = op_msg(wl)
    del bad["delta"]
    with pytest.raises(PlatformError) as ei:
        validate_message(bad, wl)
    assert ei.value.code == E_FIELD
    assert ei.value.detail["path"] == "$.delta"
    assert ei.value.detail["wl"] == wl

    with pytest.raises(PlatformError) as ei:
        validate_message(op_msg(wl, bogus=1), wl)
    assert ei.value.code == E_FIELD
    assert ei.value.detail["path"] == "$.bogus"

    with pytest.raises(PlatformError) as ei:
        validate_message(op_msg(wl, delta="x"), wl)
    assert ei.value.code == E_FIELD
    assert ei.value.detail["path"] == "$.delta"

    with pytest.raises(PlatformError) as ei:
        validate_message(op_msg("wl1.999"), wl)
    assert ei.value.code == E_FIELD
    assert ei.value.detail["path"] == "$.wl"
    assert ei.value.detail["expected"] == wl

    result = {"type": "op_result", "wl": wl, "lease_id": "L1", "node_id": "n0",
              "status": "done", "progress": 2,
              "summary": {"lease_id": "L1", "sectors": ["b0#0"], "seq_lo": 1,
                          "seq_hi": 2, "value": 1.0}}
    with pytest.raises(PlatformError) as ei:
        validate_message(result, wl)
    assert ei.value.code == E_FIELD
    assert ei.value.detail["path"] == "$.summary.versions"


def test_lease_scope_seq_and_expiry_recycling():
    clock = FakeClock()
    lm = LeaseManager(store=None, clock=clock)
    lm.grant("L1", "n0", ("b0#0",), 1, 3, 1000)
    lm.activate("L1")

    with pytest.raises(PlatformError) as ei:
        lm.check_access("L1", ["b0#1"], 1)
    assert ei.value.code == E_SCOPE
    assert ei.value.detail["sector"] == "b0#1"

    with pytest.raises(PlatformError) as ei:
        lm.check_access("L1", ["b0#0"], 5)
    assert ei.value.code == E_SEQ
    assert ei.value.detail["seq"] == 5

    lm.check_access("L1", ["b0#0"], 2)

    with pytest.raises(PlatformError) as ei:
        lm.grant("L2", "n1", ("b0#0",), 4, 5, 1000)
    assert ei.value.code == E_BUSY

    clock.advance(1.5)
    assert lm.reap() == ["L1"]
    assert lm.cached("L1") is None
    with pytest.raises(PlatformError) as ei:
        lm.check_access("L1", ["b0#0"], 2)
    assert ei.value.code == E_SCOPE

    reasons = [(t["from"], t["to"], t["reason"]) for t in lm.transitions]
    assert ("none", "granted", "granted") in reasons
    assert ("granted", "active", "node_ack") in reasons
    assert ("active", "expired", "ttl_expired") in reasons
    expiry_mono = lm.transitions[-1]["mono"]

    clock.advance(0.01)
    lease2 = lm.grant("L2", "n1", ("b0#0",), 4, 5, 1000)
    assert lease2.grant_mono >= expiry_mono


def test_owner_rejection_has_no_side_effects(tmp_path):
    p = make_platform(tmp_path)
    before = {s: p.mirror[s].copy() for s in p.sectors}
    req = Request("r1", "n1", "write", "sum", ("b0#0",), delta=1.0, count=2)
    with pytest.raises(PlatformError) as ei:
        p.submit_and_wait(req, timeout=10)
    assert ei.value.code == E_OWNER
    assert ei.value.detail["sector"] == "b0#0"
    assert all(p.versions[s] == 0 for s in p.sectors)
    assert p.leases.all_leases() == []
    for s in p.sectors:
        assert list(p.mirror[s]) == list(before[s])
    rows = p.store.list_adjudications()
    assert rows[0]["result"] == "rejected"
    assert rows[0]["detail"]["code"] == E_OWNER
    p.close()


def test_scope_rejection_on_unknown_sector(tmp_path):
    p = make_platform(tmp_path)
    req = Request("r2", "n0", "read", "sum", ("b9#9",), count=1)
    with pytest.raises(PlatformError) as ei:
        p.submit_and_wait(req, timeout=10)
    assert ei.value.code == E_SCOPE
    p.close()


def test_execute_scope_and_seq_violations(tmp_path):
    def wide_scope(platform, lease, req):
        s = platform._honest_run(lease, req)
        s["sectors"] = list(lease.sectors) + ["b1#1"]
        return s

    p = make_platform(tmp_path, node_runner=wide_scope)
    req = Request("r3", "n0", "read", "sum", ("b0#0",), count=1)
    with pytest.raises(PlatformError) as ei:
        p.submit_and_wait(req, timeout=10)
    assert ei.value.code == E_SCOPE
    p.close()

    def bad_seq(platform, lease, req):
        s = platform._honest_run(lease, req)
        s["seq_hi"] = lease.seq_hi + 1
        return s

    p = make_platform(tmp_path, name="t2.db", node_runner=bad_seq)
    req = Request("r4", "n0", "read", "sum", ("b0#0",), count=1)
    with pytest.raises(PlatformError) as ei:
        p.submit_and_wait(req, timeout=10)
    assert ei.value.code == E_SEQ
    p.close()


def test_ver_mismatch_reports_deviation(tmp_path):
    def corrupt(platform, lease, req):
        s = platform._honest_run(lease, req)
        s["value"] += 0.5
        return s

    p = make_platform(tmp_path, node_runner=corrupt)
    req = Request("r5", "n0", "read", "sum", ("b0#0",), count=2)
    with pytest.raises(PlatformError) as ei:
        p.submit_and_wait(req, timeout=10)
    assert ei.value.code == E_VER
    assert ei.value.detail["node"] == "n0"
    assert ei.value.detail["stage"] == "value"
    assert "lease" in ei.value.detail
    lease = p.leases.all_leases()[0]
    assert lease.state == "revoked"
    batches = p.store.list_batches()
    assert batches and batches[0]["match"] is False
    p.close()


def test_budget_queueing_and_global_limit(tmp_path):
    def slow(platform, lease, req):
        time.sleep(0.05)
        return platform._honest_run(lease, req)

    p = make_platform(tmp_path, node_runner=slow, global_limit=2, node_quota=1)
    reqs = [Request(f"q{i}", f"n{i % 2}", "read", "sum",
                    (f"b0#{i % 2}",), count=1) for i in range(6)]
    futs = [p.submit(r) for r in reqs]
    results = [f.result(timeout=30) for f in futs]
    assert len(results) == 6
    assert p.adjudicator.max_inflight <= 2
    assert p.adjudicator.node_max_inflight["n0"] <= 1
    assert p.adjudicator.node_max_inflight["n1"] <= 1
    granted = [r for r in p.store.list_adjudications() if r["result"] == "granted"]
    assert len(granted) == 6
    p.close()


def test_adjudication_timeout_retry_same_scope(tmp_path):
    def slow(platform, lease, req):
        time.sleep(0.35)
        return platform._honest_run(lease, req)

    p = make_platform(tmp_path, node_runner=slow, node_quota=1,
                      adjudication_timeout_s=0.2, adjudication_retries=3)
    req_a = Request("a", "n0", "read", "sum", ("b0#0",), count=1)
    req_b = Request("b", "n0", "read", "sum", ("b1#0",), count=1)
    fut_a = p.submit(req_a)
    time.sleep(0.1)
    res_b = p.submit_and_wait(req_b, timeout=30)
    fut_a.result(timeout=30)
    assert res_b["sectors"] == ["b1#0"]
    rows = [r for r in p.store.list_adjudications() if r["req_id"] == "b"]
    assert any(r["result"] == "timeout" for r in rows)
    assert rows[-1]["result"] == "granted"
    scopes = {tuple(r["sectors"]) for r in rows}
    assert scopes == {("b1#0",)}
    attempts = [r["attempt"] for r in rows]
    assert attempts == list(range(len(attempts)))
    p.close()


def run_batch_ok(p, req):
    return p.submit_and_wait(req, timeout=30)


def test_recovery_after_forced_kill(tmp_path):
    db = tmp_path / "kill.db"
    p1 = Platform(make_cfg(), db_path=str(db))
    res = run_batch_ok(p1, Request("w1", "n0", "write", "sum", ("b0#0",),
                                   delta=0.5, count=2))
    assert p1.versions["b0#0"] == 2
    dangling = p1.grant_lease(Request("d1", "n1", "read", "sum", ("b0#1",),
                                      count=1))
    p1.adjudicator.shutdown()
    p1.store.close()

    p2 = Platform(make_cfg(), db_path=str(db))
    report = p2.recover()
    assert report["ok"] is True
    assert dangling.lease_id in report["unknown_leases"]
    lease = p2.leases.get(dangling.lease_id)
    assert lease.state == "unknown"
    assert p2.versions["b0#0"] == 2
    replay = p2.replay()
    assert replay["status"] == "ok" and replay["replayed"] == 1
    p2.close()


def test_recover_detects_version_mismatch(tmp_path):
    db = tmp_path / "mm.db"
    p1 = Platform(make_cfg(), db_path=str(db))
    run_batch_ok(p1, Request("w1", "n0", "write", "sum", ("b0#0",),
                             delta=0.5, count=2))
    p1.adjudicator.shutdown()
    p1.store.close()
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE sector_versions SET version=99 WHERE sector_id='b0#0'")
    conn.commit()
    conn.close()

    p2 = Platform(make_cfg(), db_path=str(db))
    report = p2.recover()
    assert report["ok"] is False
    err = report["errors"][0]
    assert err["code"] == E_RECOVER
    assert err["sector"] == "b0#0"
    assert err["stored_version"] == 99
    assert err["expected_version"] == 2
    p2.close()


def test_snapshot_restore_wholesale_and_replay(tmp_path):
    db = tmp_path / "snap.db"
    p1 = Platform(make_cfg(), db_path=str(db))
    run_batch_ok(p1, Request("w1", "n0", "write", "sum", ("b0#0",),
                             delta=0.25, count=2))
    run_batch_ok(p1, Request("r1", "n1", "read", "mean", ("b0#1",), count=1))
    snap = p1.snapshot()
    wl1 = p1.wl
    p1.adjudicator.shutdown()
    p1.store.close()

    p2 = Platform(make_cfg(), db_path=str(db))
    assert p2.wl != wl1
    res = p2.restore_and_replay()
    assert res["wholesale_replace"] is True
    assert res["snapshot_wl"] == wl1
    assert res["replay"]["status"] == "ok"
    assert res["replay"]["replayed"] == 2
    assert p2.versions["b0#0"] == 2
    p2.close()


def test_replay_divergence_locates_first_mismatch(tmp_path):
    db = tmp_path / "div.db"
    p1 = Platform(make_cfg(), db_path=str(db))
    r1 = run_batch_ok(p1, Request("w1", "n0", "write", "sum", ("b0#0",),
                                  delta=0.25, count=2))
    run_batch_ok(p1, Request("w2", "n1", "write", "sum", ("b0#1",),
                             delta=0.5, count=1))
    p1.adjudicator.shutdown()
    p1.store.close()
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE batch_results SET value=value+1.0 WHERE lease_id=?",
                 (r1["lease_id"],))
    conn.commit()
    conn.close()

    p2 = Platform(make_cfg(), db_path=str(db))
    with pytest.raises(PlatformError) as ei:
        p2.replay()
    assert ei.value.code == E_REPLAY
    assert ei.value.detail["lease"] == r1["lease_id"]
    assert ei.value.detail["field"] == "value"
    assert ei.value.detail["sector"] == "b0#0"
    p2.close()
