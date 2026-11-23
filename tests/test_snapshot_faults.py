import json
import shutil

import pytest

from shardbench.engine import WorkbenchEngine
from shardbench.errors import BizError, E_CONFIG, E_LEASE, E_SNAPSHOT

CFG_V1 = "samples/sharding_config.json"
CFG_V2 = "samples/sharding_config_v2.json"
PRED = {"eq": ["user_id", 42]}


@pytest.fixture
def engine(tmp_path):
    eng = WorkbenchEngine(str(tmp_path), lease_ttl_ms=60_000)
    eng.load_config(CFG_V1)
    yield eng
    eng.shutdown()


def test_snapshot_roundtrip_and_recovery(engine):
    engine.compile("orders_by_user", PRED)
    engine.probe([engine.compile("orders_by_user", PRED)])
    snap = engine.save_snapshot()
    assert snap["payload"]["rule_version"] == 1
    assert snap["payload"]["ledger_cursor"] == 1
    result = engine.recover()
    assert result["recovered_cursor"] == 1
    assert result["rerun_required"] is False


def test_snapshot_version_mismatch_rejected(engine):
    engine.save_snapshot()
    engine.load_config(CFG_V2)  # bumps rule version to 2
    with pytest.raises(BizError) as ei:
        engine.recover()
    assert ei.value.code == E_SNAPSHOT
    assert ei.value.field == "rule_version"


def test_snapshot_digest_tamper_rejected(engine, tmp_path):
    engine.save_snapshot()
    path = engine.snapshot_path()
    doc = json.loads(open(path, encoding="utf-8").read())
    doc["payload"]["ledger_cursor"] = 999  # tamper without fixing digest
    open(path, "w", encoding="utf-8").write(json.dumps(doc))
    with pytest.raises(BizError) as ei:
        engine.recover()
    assert ei.value.code == E_SNAPSHOT
    assert ei.value.field == "content_digest"


def test_crash_recovery_marks_uncommitted_leases_unknown(engine, tmp_path):
    lease = engine.lease_manager.acquire(db=0)  # never released -> "held"
    engine.ledger.record_lease(lease.as_dict(), 1, 1)
    engine.save_snapshot()
    # simulate forced kill: build a fresh engine on the same workdir
    eng2 = WorkbenchEngine(str(tmp_path), lease_ttl_ms=60_000)
    try:
        eng2.load_config(CFG_V1)
        result = eng2.recover()
        assert lease.lease_id in result["unknown_leases"]
        assert result["rerun_required"] is True
        states = {l["lease_id"]: l["state"] for l in eng2.ledger.current_leases()}
        assert states[lease.lease_id] == "unknown"
    finally:
        eng2.shutdown()


def test_only_committed_batches_survive_kill(engine, tmp_path):
    plan = engine.compile("orders_by_user", PRED)
    engine.probe([plan])                       # committed batch 1
    engine.ledger.begin_batch()                # open but never commit
    engine.ledger.buffer_task(
        {"unit_id": "orders_by_user", "predicate_digest": "x", "plan_digest": "y",
         "targets": [], "reasons": [], "lease": {}, "wait_ms": 0, "exec_ms": 0,
         "status": "ok"},
        engine.versions.current())
    eng2 = WorkbenchEngine(str(tmp_path), lease_ttl_ms=60_000)
    try:
        eng2.load_config(CFG_V1)
        assert eng2.ledger.last_committed_batch() == 1
        assert len(eng2.ledger.current_tasks()) == len(plan.targets)
    finally:
        eng2.shutdown()


def test_fault_injection_corrupt_config(engine, tmp_path):
    bad = tmp_path / "corrupted.json"
    shutil.copy(CFG_V1, bad)
    err = engine.inject_config_corruption(str(bad), mode="invalid_json")
    assert err.code == E_CONFIG
    err2 = None
    shutil.copy(CFG_V1, bad)
    err2 = engine.inject_config_corruption(str(bad), mode="bad_field")
    assert err2.code == E_CONFIG
    assert err2.field == "routing_units[0].physical_dbs"


def test_fault_injection_lease_version_conflict(engine):
    lease = engine.lease_manager.acquire(db=1)
    evidence = engine.inject_lease_version_conflict(lease.lease_id)
    assert evidence["injected_version"] == 2
    with pytest.raises(BizError) as ei:
        engine.lease_manager.renew(lease.lease_id, expected_version=1)
    assert ei.value.code == E_LEASE
    assert ei.value.detail["current_version"] == 2


def test_faults_written_to_structured_log(engine, tmp_path):
    bad = tmp_path / "c.json"
    shutil.copy(CFG_V1, bad)
    engine.inject_config_corruption(str(bad))
    lease = engine.lease_manager.acquire(db=0)
    engine.inject_lease_version_conflict(lease.lease_id)
    log_path = engine.logger.path
    lines = [json.loads(l) for l in open(log_path, encoding="utf-8")]
    kinds = [l.get("kind") for l in lines if l["event"] == "fault_injected"]
    assert kinds == ["config_corruption", "lease_version_conflict"]
