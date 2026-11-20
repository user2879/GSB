#!/usr/bin/env python3
"""端到端验收演示：三路径一致性、默认路径选择、强制终止恢复、三类异常码。

运行：python demo.py
所有产物写入 samples/demo/，结构化日志写入 logs/demo-events.jsonl。
"""
from __future__ import annotations

import json
import os

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
    choose_and_describe,
    persist_run,
    recover_and_rerun,
    run_all_paths,
    spec_digest,
)
from quota_arbiter.snapshot import read_snapshot, verify_replay
from quota_arbiter.util import StructuredLogger

WORKDIR = "samples/demo"
LOG = "logs/demo-events.jsonl"


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    logger = StructuredLogger(LOG)
    report: dict = {"steps": []}

    generate_dataset(WORKDIR, tenant_count=10, pool_count=3, request_count=600)
    tenants, pools = load_spec(f"{WORKDIR}/spec.json")
    requests = load_requests(f"{WORKDIR}/requests.ndjson", tenants, pools)

    # 1) 默认路径选择
    selection = choose_and_describe(tenants, pools, requests)
    report["steps"].append({"name": "default_path_selection", "selection": selection})

    # 2) 三路径逐字节一致性
    results, consistency = run_all_paths(tenants, pools, requests)
    report["steps"].append({"name": "three_path_consistency",
                            "identical": consistency["identical"],
                            "outcome_bytes": consistency["outcome_bytes"],
                            "state_bytes": consistency["state_bytes"],
                            "trace_bytes": consistency["trace_bytes"]})
    assert consistency["identical"]

    # 3) 落账 + 快照重放
    digest = spec_digest(tenants, pools, requests)
    ledger_path = f"{WORKDIR}/ledger.sqlite3"
    if os.path.exists(ledger_path):
        os.remove(ledger_path)
    ledger = Ledger(ledger_path)
    ledger.initialize_session(tenants, pools, selection["selected"], digest)
    persist_run(tenants, pools, requests, selection["selected"], ledger, logger=logger)
    ledger.verify_digest_chain()
    ledger.close()

    result = run_path(selection["selected"], tenants, pools, requests)
    from quota_arbiter.snapshot import write_snapshot

    snapshot_path = f"{WORKDIR}/snapshot.json"
    write_snapshot(snapshot_path, tenants, pools, selection, result, result.last_rid)
    payload = read_snapshot(snapshot_path)
    replay = run_path(payload["default_path"], tenants, pools, requests)
    verify_replay(payload, replay, payload["processed_cursor"])
    report["steps"].append({"name": "snapshot_replay", "replay_verified": True})

    # 4) 强制终止恢复
    crash_path = f"{WORKDIR}/crash.sqlite3"
    if os.path.exists(crash_path):
        os.remove(crash_path)
    crash_ledger = Ledger(crash_path)
    crash_ledger.initialize_session(tenants, pools, selection["selected"], digest)
    try:
        persist_run(tenants, pools, requests, selection["selected"],
                    crash_ledger, crash_after_batch=2, logger=logger)
    except Exception:
        crash_ledger.close()
    point = recover_and_rerun(
        tenants, pools, requests, Ledger(crash_path), digest,
        selection["selected"], logger=logger,
    )
    report["steps"].append({
        "name": "forced_termination_recovery",
        "last_committed_rid": point["last_committed_rid"],
        "unknown_count": point["unknown_count"],
        "rerun_ledger": point["rerun_ledger"],
    })

    # 5) 三类异常注入业务码
    faults = {}
    order_dst = f"{WORKDIR}/fault_order.ndjson"
    inject_order_regression(f"{WORKDIR}/requests.ndjson", order_dst)
    try:
        load_requests(order_dst, tenants, pools, check_global_order=True)
    except Exception as exc:
        faults["order"] = exc.code
        logger.error("fault", **exc.to_dict(), injection="order")

    spec_dst = f"{WORKDIR}/fault_spec.json"
    inject_missing_priority_tenant(f"{WORKDIR}/spec.json", spec_dst)
    try:
        load_spec(spec_dst)
    except Exception as exc:
        faults["priority"] = exc.code
        logger.error("fault", **exc.to_dict(), injection="priority")

    tamper = Ledger(ledger_path)
    rid = tamper.conn.execute(
        "SELECT rid FROM outcomes WHERE verdict != 'REJECT' ORDER BY rid LIMIT 1"
    ).fetchone()[0]
    inject_ledger_digest(tamper, rid)
    tamper.close()
    try:
        Ledger(ledger_path).verify_digest_chain()
    except Exception as exc:
        faults["ledger"] = exc.code
        logger.error("fault", **exc.to_dict(), injection="ledger")
    report["steps"].append({"name": "fault_injection_codes", "faults": faults})

    assert faults == {"order": "E_ORDER", "priority": "E_PRIORITY", "ledger": "E_LEDGER"}
    out = f"{WORKDIR}/acceptance_report.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\n验收报告已写入 {out}，结构化日志 {LOG}")


if __name__ == "__main__":
    main()
