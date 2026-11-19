"""可重复验证入口：一条命令生成样本集并执行固定序列。

  python -m gsb.cli run --out <目录>

固定序列：生成样本 -> 三格式解析与等价性核对 -> 迁移幂等检查 -> 合成 ->
压缩 -> 追赶 -> 恢复（强制终止局）-> 独立验证。输出逐版本哈希清单。
同一命令两次运行的 report.json 逐字节相同。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from .engine import Engine, catch_up
from .equivalence import equivalence_report
from .errors import (
    E_BASE_VERSION,
    E_FORMAT_FIELD,
    E_PEER_GAP,
    E_SCHEMA_TOO_NEW,
    PipelineError,
)
from .formats import binary as fmt_binary
from .generator import SCENARIOS, generate_all
from .migrate import audit_table, migrate_event
from .model import Event
from .pipeline import load_all_formats, load_format, load_mixed
from .store import Store
from .verifier import verify

KILL_DIE_AFTER = 5


def _run_engine_scenario(scenario: str, scenario_dir: str, store_dir: str) -> dict:
    audit: list[dict] = []
    streams = load_all_formats(scenario_dir)
    eq_report = equivalence_report(streams)
    if scenario == "mixed":
        events = load_mixed(scenario_dir, audit)
    else:
        events = load_format(scenario_dir, "jsonl", audit)

    store = Store(store_dir)
    recovery = None
    if scenario == "kill":
        recovery = _run_kill_sequence(scenario_dir, store_dir, events)
        store = Store(store_dir)
    else:
        engine = Engine(store)
        engine.run(events)

    manifest = store.load_manifest()
    head = manifest["head"]
    base = manifest["versions"][0]["version"]
    catch_from = max(base, head - 3)
    catch_report = catch_up(store, catch_from, head)
    verify_report = verify(store_dir)
    online_hashes = store.read_json("online_hashes.json")

    return {
        "equivalence": eq_report,
        "migration_audit_entries": len(audit),
        "recovery": recovery,
        "head": head,
        "compacted_base": manifest.get("compacted_base"),
        "catch_up": catch_report,
        "verify": {
            "consistent": verify_report["consistent"],
            "versions_checked": verify_report["versions_checked"],
            "first_mismatch": verify_report["first_mismatch"],
        },
        "adjudications": _count_lines(store_dir, "adjudication.jsonl"),
        "hashes": online_hashes,
    }


def _count_lines(store_dir: str, name: str) -> int:
    path = os.path.join(store_dir, name)
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _run_kill_sequence(scenario_dir: str, store_dir: str, events: list[Event]) -> dict:
    """强制终止局：子进程提交若干版本后被杀，重启恢复并续跑剩余事件。"""
    cmd = [sys.executable, "-m", "gsb.cli", "worker",
           "--scenario-dir", scenario_dir, "--store", store_dir,
           "--die-after", str(KILL_DIE_AFTER)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        raise RuntimeError("worker 未被强制终止，终止局构造失败")

    store = Store(store_dir)
    recovery = store.recover()
    head = recovery["head"] or 0
    engine = Engine.resume(store)
    remaining = [e for e in events if e.tick >= head]
    engine.run(remaining)
    recovery["resumed_from_version"] = head
    recovery["remaining_events"] = len(remaining)
    recovery["worker_returncode"] = proc.returncode
    return recovery


def _negative_checks() -> list[dict]:
    """错误码注入检查：每个业务错误码都必须按约定触发。"""
    results = []

    def record(name: str, fn, expect_code: str):
        try:
            fn()
        except PipelineError as exc:
            results.append({"check": name, "code": exc.code,
                            "offset": exc.offset, "ok": exc.code == expect_code})
            return
        results.append({"check": name, "code": None, "ok": False})

    def too_new():
        event = Event(0, "peer-1", 0, "move", {"entity": "e0", "dx": 1, "dy": 0},
                      0, schema_version=99)
        migrate_event(event)

    def bad_endian():
        frame = bytearray(fmt_binary.encode_frame(
            Event(0, "peer-1", 0, "move", {"entity": "e0", "dx": 1, "dy": 0}, 0).finalize()))
        frame[3] = 0x7F
        fmt_binary.decode_frame(bytes(frame))

    def bad_width():
        frame = bytearray(fmt_binary.encode_frame(
            Event(0, "peer-1", 0, "move", {"entity": "e0", "dx": 1, "dy": 0}, 0).finalize()))
        frame[4] = 3
        fmt_binary.decode_frame(bytes(frame))

    def trailing():
        frame = fmt_binary.encode_frame(
            Event(0, "peer-1", 0, "move", {"entity": "e0", "dx": 1, "dy": 0}, 0).finalize())
        fmt_binary.decode_frame(frame + b"\x00")

    def peer_gap():
        from .engine import Engine as _Engine
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            engine = _Engine(Store(tmp))
            engine.run([Event(0, "peer-1", 1, "move",
                              {"entity": "e0", "dx": 1, "dy": 0}, 0).finalize()])

    def base_version():
        import tempfile
        from .engine import Engine as _Engine
        with tempfile.TemporaryDirectory() as tmp:
            engine = _Engine(Store(tmp))
            engine.run([Event(0, "peer-1", 0, "move",
                              {"entity": "e0", "dx": 1, "dy": 0}, 42).finalize()])

    record("schema_too_new", too_new, E_SCHEMA_TOO_NEW)
    record("binary_bad_endian", bad_endian, E_FORMAT_FIELD)
    record("binary_bad_index_width", bad_width, E_FORMAT_FIELD)
    record("binary_trailing_bytes", trailing, E_FORMAT_FIELD)
    record("peer_gap", peer_gap, E_PEER_GAP)
    record("unknown_base_version", base_version, E_BASE_VERSION)
    return results


def _migration_idempotency_check(scenario_dir: str) -> dict:
    """对同一事件重复迁移，内容哈希必须保持不变。"""
    events = load_format(scenario_dir, "jsonl")
    checked = 0
    for event in events[:20]:
        first = event.compute_event_id()
        again, audit = migrate_event(Event.from_dict(event.to_dict()))
        again.finalize()
        if again.compute_event_id() != first or audit:
            return {"idempotent": False, "checked": checked}
        checked += 1
    return {"idempotent": True, "checked": checked}


def cmd_run(out_dir: str) -> int:
    scenarios_dir = os.path.join(out_dir, "scenarios")
    stores_dir = os.path.join(out_dir, "stores")
    os.makedirs(stores_dir, exist_ok=True)

    generated = generate_all(scenarios_dir)
    report: dict = {
        "seed": 250,
        "migration_table": audit_table(),
        "scenarios": {},
        "negative_checks": _negative_checks(),
    }
    for scenario in SCENARIOS:
        scenario_dir = os.path.join(scenarios_dir, scenario)
        store_dir = os.path.join(stores_dir, scenario)
        report["scenarios"][scenario] = _run_engine_scenario(
            scenario, scenario_dir, store_dir)
        report["scenarios"][scenario]["event_count"] = generated[scenario]["event_count"]
    report["migration_idempotency"] = _migration_idempotency_check(
        os.path.join(scenarios_dir, "upgrade"))

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, sort_keys=True, ensure_ascii=False, indent=1)
        fh.write("\n")

    for scenario in SCENARIOS:
        entry = report["scenarios"][scenario]
        print(f"[{scenario}] head=v{entry['head']} "
              f"equivalence={entry['equivalence']['all_equal']} "
              f"verify={entry['verify']['consistent']} "
              f"catch_up={entry['catch_up']['match']}")
        for h in entry["hashes"]:
            print(f"  v{h['version']:>2}  {h['state_hash']}")
    print(f"report: {report_path}")
    return 0


def cmd_worker(scenario_dir: str, store_dir: str, die_after: int) -> int:
    events = load_format(scenario_dir, "jsonl")
    engine = Engine(Store(store_dir), die_after_commits=die_after)
    engine.run(events)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gsb")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="生成样本集并执行固定验证序列")
    p_run.add_argument("--out", required=True)
    p_worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    p_worker.add_argument("--scenario-dir", required=True)
    p_worker.add_argument("--store", required=True)
    p_worker.add_argument("--die-after", type=int, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        return cmd_run(args.out)
    if args.cmd == "worker":
        return cmd_worker(args.scenario_dir, args.store, args.die_after)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
