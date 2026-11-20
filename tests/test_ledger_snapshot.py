"""SQLite 账本原子提交、强制终止恢复、快照重放与 E_LEDGER。"""
import os

import pytest

from quota_arbiter.errors import LedgerError, ReplayError
from quota_arbiter.generator import (
    generate_dataset,
    inject_ledger_digest,
    inject_missing_priority_tenant,
    inject_order_regression,
)
from quota_arbiter.ledger import Ledger
from quota_arbiter.loader import load_requests, load_spec
from quota_arbiter.paths import run_path
from quota_arbiter.runner import (
    persist_run,
    recover_and_rerun,
    choose_and_describe,
    snapshot_and_replay,
    spec_digest,
)
from quota_arbiter.snapshot import read_snapshot, restore_snapshot


@pytest.fixture()
def dataset(tmp_path):
    generate_dataset(str(tmp_path), tenant_count=6, pool_count=2, request_count=240)
    workdir = str(tmp_path)
    tenants, pools = load_spec(f"{workdir}/spec.json")
    requests = load_requests(f"{workdir}/requests.ndjson", tenants, pools)
    return workdir, tenants, pools, requests


def test_batch_atomic_commit_and_digest_chain(dataset):
    workdir, tenants, pools, requests = dataset
    digest = spec_digest(tenants, pools, requests)
    ledger = Ledger(f"{workdir}/ledger.sqlite3")
    ledger.initialize_session(tenants, pools, "batch", digest)
    persist_run(tenants, pools, requests, "batch", ledger)
    head = ledger.verify_digest_chain()
    assert len(head) == 64
    count = ledger.conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    assert count == len(requests)
    ledger.close()


def test_forced_termination_marks_unknown_and_reruns(dataset):
    workdir, tenants, pools, requests = dataset
    digest = spec_digest(tenants, pools, requests)
    selection = choose_and_describe(tenants, pools, requests)
    ledger_path = f"{workdir}/ledger.sqlite3"
    ledger = Ledger(ledger_path)
    ledger.initialize_session(tenants, pools, selection["selected"], digest)
    with pytest.raises(LedgerError):
        persist_run(
            tenants, pools, requests, selection["selected"], ledger,
            crash_after_batch=1,
        )
    # 原连接已随崩溃关闭；重新打开只能看到完整提交的批次
    recovered = Ledger(ledger_path)
    point = recover_and_rerun(
        tenants, pools, requests, recovered, digest, selection["selected"]
    )
    assert point["unknown_count"] > 0
    assert point["last_committed_rid"] is not None
    assert os.path.exists(point["rerun_ledger"])
    # 重跑账本的摘要链有效且覆盖全部请求
    rerun = Ledger(point["rerun_ledger"])
    rerun.verify_digest_chain()
    total = rerun.conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
    assert total == len(requests)


def test_ledger_tamper_detected_e_ledger(dataset):
    workdir, tenants, pools, requests = dataset
    digest = spec_digest(tenants, pools, requests)
    path = f"{workdir}/ledger.sqlite3"
    ledger = Ledger(path)
    ledger.initialize_session(tenants, pools, "realtime", digest)
    persist_run(tenants, pools, requests, "realtime", ledger)
    ledger.close()

    damaged = Ledger(path)
    rid = damaged.conn.execute(
        "SELECT rid FROM outcomes WHERE verdict != 'REJECT' ORDER BY rid LIMIT 1"
    ).fetchone()[0]
    inject_ledger_digest(damaged, rid)
    damaged.close()

    with pytest.raises(LedgerError) as exc:
        Ledger(path).verify_digest_chain()
    assert exc.value.code == "E_LEDGER"
    assert exc.value.ref_id == rid


def test_snapshot_write_restore_and_replay(dataset):
    workdir, tenants, pools, requests = dataset
    selection = choose_and_describe(tenants, pools, requests)
    result = run_path(selection["selected"], tenants, pools, requests)
    snapshot_path = f"{workdir}/snapshot.json"
    snapshot_and_replay(tenants, pools, requests, selection, result, snapshot_path)
    payload = read_snapshot(snapshot_path)
    assert payload["format_version"] == 1
    payload2, mismatch, backup = restore_snapshot(snapshot_path, tenants, pools)
    assert mismatch is False and backup is None
    assert payload2["content_digest"] == payload["content_digest"]


def test_snapshot_mismatch_full_replace_with_readonly_backup(dataset):
    workdir, tenants, pools, requests = dataset
    selection = choose_and_describe(tenants, pools, requests)
    result = run_path(selection["selected"], tenants, pools, requests)
    snapshot_path = f"{workdir}/snapshot.json"
    snapshot_and_replay(tenants, pools, requests, selection, result, snapshot_path)

    payload = read_snapshot(snapshot_path)
    payload["tenant_set_version"] = "0" * 64
    from quota_arbiter.util import canonical_json, digest_of

    payload["content_digest"] = digest_of(
        {k: v for k, v in payload.items() if k != "content_digest"}
    )
    alt = f"{workdir}/alt.json"
    with open(alt, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(payload))
    _, mismatch, backup = restore_snapshot(alt, tenants, pools)
    assert mismatch is True
    assert backup == alt + ".readonly.bak"
    assert not os.access(backup, os.W_OK)


def test_corrupt_snapshot_raises_e_replay(dataset):
    workdir, tenants, pools, requests = dataset
    selection = choose_and_describe(tenants, pools, requests)
    result = run_path(selection["selected"], tenants, pools, requests)
    snapshot_path = f"{workdir}/snapshot.json"
    snapshot_and_replay(tenants, pools, requests, selection, result, snapshot_path)
    payload = read_snapshot(snapshot_path)
    payload["content_digest"] = "f" * 64
    bad = f"{workdir}/bad.json"
    from quota_arbiter.util import canonical_json

    with open(bad, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(payload))
    with pytest.raises(ReplayError) as exc:
        read_snapshot(bad)
    assert exc.value.code == "E_REPLAY"


def test_injection_helpers_only_change_data(dataset):
    workdir, tenants, pools, requests = dataset
    order_dst = f"{workdir}/order_fault.ndjson"
    meta = inject_order_regression(f"{workdir}/requests.ndjson", order_dst)
    assert os.path.exists(order_dst) and meta["field"] == "arrival"

    spec_dst = f"{workdir}/spec_fault.json"
    meta = inject_missing_priority_tenant(f"{workdir}/spec.json", spec_dst)
    assert meta["missing_tid"] == "T_MISSING_9999"
