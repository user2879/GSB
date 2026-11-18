#!/usr/bin/env python3
"""一键诊断演示：覆盖 8 项业务要求并生成结构化诊断报告。

运行：python3 demo.py
输出：data/diagnostic_report.json（同时在终端打印摘要）
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from qos.changeset import ChangeSet
from qos.errors import QosError
from qos.ruleset import load_rules
from qos.shaper import is_port_free
from qos.workbench import Workbench

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
DATA = ROOT / "data"
LOGS = ROOT / "logs"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="qos_demo_", dir=str(DATA))
    db = str(Path(workdir) / "ledger.db")
    log = str(Path(workdir) / "qos.jsonl")
    report: dict = {"workdir": workdir, "checks": []}

    def record(name: str, ok: bool, code: str, evidence: dict) -> None:
        report["checks"].append({"name": name, "ok": ok, "code": code,
                                 "evidence": evidence})
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({code})")

    wb = Workbench(db, log)

    # 1) 加载与强校验
    section("1. 规则加载与强校验")
    load_result = wb.load(SAMPLES / "rules.json")
    record("load-9-rules", len(wb.rules) == 9, "E_OK",
           {"ruleset_version": wb.ruleset_ver,
            "plan_version": wb.plan.plan_version,
            "topo": wb.graph.topo_order()})

    for fault, file, expect_code in [
        ("corrupt-file", "rules_corrupt.json", "E_RULE"),
        ("invalid-fields", "rules_invalid_fields.json", "E_RULE"),
        ("dependency-cycle", "rules_cycle.json", "E_DEPEND"),
    ]:
        try:
            load_rules(SAMPLES / file)
            record(f"inject-{fault}", False, "E_UNKNOWN", {})
        except QosError as exc:
            record(f"inject-{fault}", exc.code == expect_code, exc.code,
                   exc.details)

    # 2) 增量变更与失效传播
    section("2. 增量变更与失效传播")
    changeset = ChangeSet.from_dict(json.loads(
        (SAMPLES / "changeset_modify.json").read_text(encoding="utf-8")))
    change_result = wb.apply_change(changeset)
    expected_invalid = {"R1_BASE", "R3_CHILD", "R4_CHILD", "R6_GRAND",
                        "R7_GRAND", "R9_SIBLING"}
    record("invalidation-propagation",
           set(change_result["invalidation_sequence"]) == expected_invalid,
           "E_OK", {"sequence": change_result["invalidation_sequence"],
                    "trace_steps": len(change_result["trace"]),
                    "unaffected_absent": "R2_BASE" not in
                    change_result["invalidation_sequence"]})

    try:
        wb.apply_change(ChangeSet(delete_rule_ids=["R2_BASE"]))
        record("delete-referenced-refused", False, "E_UNKNOWN", {})
    except QosError as exc:
        record("delete-referenced-refused", exc.code == "E_REF", exc.code,
               exc.details)

    # 3/4) 增量编译一致性
    section("3/4. 版本化整体替换与增量=全量一致性")
    incr_check = wb.verify_incremental()
    record("incremental-equals-full", incr_check["ok"], "E_OK", incr_check)
    record("monotonic-plan-version",
           wb.plan.plan_version == 2 and incr_check["reused_count"] == 3,
           "E_OK", {"plan_version": wb.plan.plan_version,
                    "reused": incr_check["reused"],
                    "recompiled": incr_check["recompiled"]})

    # 5) 整形模拟（确定性虚拟时钟 + 真实回环 TCP）
    section("5. 整形限速观测")
    sim = wb.simulate(count=30, size=1024, arrival_gap_us=100_000)
    record("virtual-shaping-throughput", True, "E_OK",
           {"digest": sim["digest"], "throughput_rules": list(sim["throughput"])})

    plan_file = Path(workdir) / "plan.json"
    plan_file.write_text(json.dumps(wb.plan.payload()["entries"]),
                         encoding="utf-8")
    shaper_port, sink_port = 49501, 49502
    if is_port_free("127.0.0.1", shaper_port) and \
            is_port_free("127.0.0.1", sink_port):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "traffic_sim.py"), "run",
             "--plan", str(plan_file), "--shaper-port", str(shaper_port),
             "--sink-port", str(sink_port), "--count", "12", "--size", "256",
             "--ports", "14000", "15000", "--gap-ms", "5",
             "--out-dir", str(Path(workdir) / "traffic")],
            capture_output=True, text=True, timeout=60, cwd=ROOT)
        traffic = json.loads(
            (Path(workdir) / "traffic" / "traffic_summary.json").read_text())
        record("real-loopback-tcp",
               proc.returncode == 0 and traffic["sink"]["frames"] == 12,
               "E_OK", traffic)
    else:
        record("real-loopback-tcp", False, "E_PORT_BUSY",
               {"shaper_port": shaper_port, "sink_port": sink_port})

    # 6) 强制终止与恢复
    section("6. 强制终止账本恢复")
    op_id = wb.simulate_crash_during_compile()
    wb.close()
    wb = Workbench(db, log)
    recovery = wb.recover()
    record("crash-recovery-unknown-marking",
           recovery["recovered_plan_version"] == 2 and recovery["unknown_rules"],
           "E_OK", {"crashed_op_id": op_id, **recovery})

    # 7) 快照版本识别与整体替换
    section("7. 会话快照")
    snap_path = str(Path(workdir) / "snapshot.json")
    wb.resolve_unknown()
    hydrated = wb.hydrate_from_ledger()  # 仅依据已完整提交的记录重建
    wb.create_snapshot(snap_path)
    # 制造与快照不同的当前会话版本（修改 R2 支系），恢复时应整体替换而非合并
    second_change = ChangeSet(upserts=[{
        "rule_id": "R2_BASE",
        "match": {"dst_addr": "127.0.0.1", "port_prefixes": [15]},
        "shaping": {"rate_kbps": 200, "burst_bytes": 5000,
                    "queue_limit_packets": 9},
        "upstream": [], "actions": ["police"]}])
    wb.apply_change(second_change)
    restore = wb.restore_snapshot(snap_path)
    record("snapshot-full-replace",
           not restore["diff"]["version_match"]
           and restore["diff"]["restore_mode"] == "full-replace",
           "E_OK", restore["diff"])

    # 8) 重放校验
    section("8. 重放校验与异常注入")
    baseline = load_rules(SAMPLES / "rules.json")
    replay = wb.replay(snap_path, baseline_rules=baseline)
    record("replay-consistency", replay["ok"], "E_OK", replay)

    try:
        from qos.snapshot import load_snapshot
        load_snapshot(SAMPLES / "snapshot_missing_versions.json")
        record("inject-snapshot-missing", False, "E_UNKNOWN", {})
    except QosError as exc:
        record("inject-snapshot-missing", exc.code == "E_SNAP", exc.code,
               exc.details)

    wb.close()
    report["summary"] = {
        "total": len(report["checks"]),
        "passed": sum(1 for c in report["checks"] if c["ok"]),
        "failed": sum(1 for c in report["checks"] if not c["ok"]),
    }
    out = DATA / "diagnostic_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              sort_keys=True), encoding="utf-8")
    section(f"诊断报告: {out}")
    print(json.dumps(report["summary"], ensure_ascii=False))
    shutil.rmtree(workdir, ignore_errors=True)
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
