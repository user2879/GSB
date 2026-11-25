#!/usr/bin/env python3
"""演示入口：覆盖回执归一、租约配额调度、失效传播与中断恢复。

子命令：
  validate     强校验合法/非法样本（E_SPEC 首个非法位置与字段）
  receipts     r1/r2/r3 编解码归一（含 E_FIELD/E_REF 拒绝）
  plan         确定性批切分与逐字节指纹
  run          真实子进程 + 回环 TCP + 双重上限调度 + 原子账本
  invalidate   参数版本/种子/批大小失效传播与子集重算对比
  crash        SIGKILL 强制终止 -> 重启恢复 -> 独立重放定位差异
  all          顺序执行上述全部场景并汇总
"""

import argparse
import copy
import glob
import json
import os
import subprocess
import sys

from batch_receipt_ledger.dashboard import make_dashboard
from batch_receipt_ledger.errors import BatchError
from batch_receipt_ledger.invalidation import plan_identity_report
from batch_receipt_ledger.plan import build_plan, plan_fingerprint
from batch_receipt_ledger.recovery import recover
from batch_receipt_ledger.receipts import encode_receipt, normalize_batch
from batch_receipt_ledger.service import run_invalidation, run_pipeline
from batch_receipt_ledger.spec import parse_spec

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(ROOT, "samples")
RUN_DIR = os.path.join(ROOT, "artifacts", "run")


def _load(name):
    path = os.path.join(SAMPLE_DIR, name)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _clean(pattern):
    for path in glob.glob(os.path.join(RUN_DIR, pattern)):
        os.remove(path)


def _kill_orphan_workers():
    """清理上一次 SIGKILL 演示遗留的工作进程（精确匹配，不波及自身）。"""
    import signal
    own = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == own:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmdline = handle.read().replace(b"\x00", b" ").decode(
                    "utf-8", "ignore")
        except OSError:
            continue
        if cmdline.strip().endswith("-m batch_receipt_ledger.worker"):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def cmd_validate(_args):
    print("== 需求1：JSON 强校验 ==")
    valid = parse_spec(_load("spec_valid.json"))
    print(f"合法样本: {len(valid.workers)} 个工作进程, "
          f"{len(valid.deliveries)} 条投递, 有效请求 {len(valid.effective_deliveries())}")
    invalid_files = sorted(name for name in os.listdir(SAMPLE_DIR)
                           if name.startswith("bad_"))
    for name in invalid_files:
        try:
            parse_spec(_load(name))
        except BatchError as exc:
            print(f"  {name}: {exc.code} location={exc.location} field={exc.field}")
    return 0


def cmd_receipts(_args):
    print("== 需求2：r1/r2/r3 编解码与归一 ==")
    spec = parse_spec(_load("spec_valid.json"))
    plans = build_plan(spec)
    plan = plans[0]
    rank_index = {rid: i for i, rid in enumerate(plan.request_ids)}
    supersession = spec.supersession_map()
    context = {
        "batch_size": spec.batch_size,
        "known_request_ids": spec.all_request_ids(),
        "worker_id": plan.worker_id,
        "param_version": 1,
        "rank_of": lambda rid: rank_index[rid],
        "dup_ref_of": lambda rid: supersession.get(rid),
    }
    r1 = [{"batch_id": plan.batch_id, "request_id": rid,
           "output_digest": f"digest-{rid}"} for rid in plan.request_ids]
    normalized = normalize_batch(r1, context)
    for norm in normalized[:1]:
        print("  r1 归一:", json.dumps(norm.to_dict(), ensure_ascii=False))
        for fmt in (1, 2, 3):
            print(f"  编码为 r{fmt}:", encode_receipt(norm, fmt))

    bad_elapsed = copy.deepcopy(r1)
    bad_elapsed[0].update({"param_version": 1, "elapsed_ms": -1,
                           "worker_id": plan.worker_id})
    try:
        normalize_batch(bad_elapsed, context)
    except BatchError as exc:
        print(f"  负耗时拒绝: {exc.code} field={exc.field}")

    bad_rank = copy.deepcopy(r1)
    for item in bad_rank:
        item.update({"param_version": 1, "elapsed_ms": 0,
                     "worker_id": plan.worker_id, "rank_in_batch": 0,
                     "dup_ref": None})
    try:
        normalize_batch(bad_rank, context)
    except BatchError as exc:
        print(f"  重复名次拒绝: {exc.code} field={exc.field}")

    bad_ref = copy.deepcopy(r1)
    bad_ref[0].update({"param_version": 1, "elapsed_ms": 0,
                       "worker_id": plan.worker_id, "rank_in_batch": 0,
                       "dup_ref": "req-ghost"})
    try:
        normalize_batch(bad_ref, context)
    except BatchError as exc:
        print(f"  悬空重复投递引用拒绝: {exc.code} field={exc.field}")
    return 0


def cmd_plan(_args):
    print("== 需求3：确定性批切分 ==")
    spec = parse_spec(_load("spec_valid.json"))
    plans = build_plan(spec)
    fingerprint = plan_fingerprint(plans)
    shuffled = copy.deepcopy(_load("spec_valid.json"))
    shuffled["workers"] = list(reversed(shuffled["workers"]))
    shuffled["deliveries"] = list(reversed(shuffled["deliveries"]))
    shuffled_plans = build_plan(parse_spec(shuffled))
    same = fingerprint == plan_fingerprint(shuffled_plans)
    for plan in plans:
        print(" ", plan.to_dict())
    print(f"  乱序声明/投递后逐字节指纹一致: {same} ({fingerprint})")
    return 0


def cmd_run(args):
    print("== 需求4/5/6/8：真实回环调度 + 租约 + 原子账本 ==")
    os.makedirs(RUN_DIR, exist_ok=True)
    _clean("demo.sqlite3*")
    spec_dict = _load("spec_valid.json")
    spec = parse_spec(spec_dict)
    dashboard, backend = make_dashboard(prefer_tk=args.gui)
    print(f"  监控面板后端: {backend}")
    outcome = run_pipeline(
        spec, spec_dict, os.path.join(RUN_DIR, "demo.sqlite3"),
        on_event=dashboard.emit)
    print("  批数:", len(outcome["results"]))
    print("  调度审计:", json.dumps(outcome["audit"], ensure_ascii=False)[:400])
    print("  重放校验:", outcome["replay"])
    return 0


def cmd_invalidate(_args):
    print("== 需求7：失效传播与仅失效子集重算 ==")
    os.makedirs(RUN_DIR, exist_ok=True)
    _clean("demo.sqlite3*; invalid*.sqlite3*")
    old_dict = _load("spec_valid.json")
    old_spec = parse_spec(old_dict)
    old_outcome = run_pipeline(
        old_spec, old_dict, os.path.join(RUN_DIR, "invalid_base.sqlite3"))

    new_dict = _load("spec_param_v2.json")
    new_spec = parse_spec(new_dict)
    identity = plan_identity_report(old_spec, new_spec)
    report = run_invalidation(
        old_spec, old_dict, new_spec, new_dict, old_outcome,
        os.path.join(RUN_DIR, "invalid_v2.sqlite3"))
    print("  变更:", identity["changes"])
    print("  失效步骤:")
    for step in report["steps"]:
        print("   ", step)
    print("  未失效批摘要保持:", report["reused_batches"])
    print("  与全量重算逐项比较:",
          report["comparison_with_full_recompute"]["all_equal"],
          f"匹配 {report['comparison_with_full_recompute']['matched']} 项")

    seed_dict = copy.deepcopy(old_dict)
    seed_dict["model"]["seed"] += 1
    seed_spec = parse_spec(seed_dict)
    seed_report = run_invalidation(
        old_spec, old_dict, seed_spec, seed_dict, old_outcome,
        os.path.join(RUN_DIR, "invalid_seed.sqlite3"))
    print("  种子变更失效批数:",
          len(seed_report["invalidated_batches"]),
          "全量重算一致:",
          seed_report["comparison_with_full_recompute"]["all_equal"])
    return 0


def cmd_crash(_args):
    print("== 需求8：强制终止后恢复 ==")
    os.makedirs(RUN_DIR, exist_ok=True)
    _clean("crash.sqlite3*")
    _kill_orphan_workers()
    db_path = os.path.join(RUN_DIR, "crash.sqlite3")
    spec_dict = _load("spec_valid.json")
    env = dict(os.environ)
    env["BRL_DEMO_DB"] = db_path
    proc = subprocess.run(
        [sys.executable, "-m", "batch_receipt_ledger.crash_cli"],
        input=json.dumps(spec_dict), env=env, capture_output=True, text=True,
        timeout=60)
    print(proc.stdout.strip())
    print("  子进程退出码（SIGKILL 期望为负9）:", proc.returncode)

    from batch_receipt_ledger.ledger import Ledger
    pre_ledger = Ledger(db_path)
    tampered = pre_ledger.conn.execute(
        "SELECT snapshot_version, batch_id FROM receipts "
        "WHERE fmt=1 LIMIT 1").fetchone()
    pre_ledger.conn.execute(
        "UPDATE receipts SET payload=? WHERE snapshot_version=? AND batch_id=? "
        "AND fmt=1",
        ('{"tampered": true}', tampered[0], tampered[1]))
    pre_ledger.conn.commit()
    pre_ledger.close()

    spec = parse_spec(spec_dict)
    report = recover(db_path, spec)
    print("  恢复结果:", json.dumps(report, ensure_ascii=False, indent=2))

    ledger = Ledger(db_path)
    diff = ledger.replay_verify()
    ledger.close()
    print("  篡改后独立重放首个差异:", json.dumps(diff, ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="本地批推理调度与回执演示")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("receipts")
    sub.add_parser("plan")
    run_p = sub.add_parser("run")
    run_p.add_argument("--gui", action="store_true", help="优先使用 Tk 面板")
    sub.add_parser("invalidate")
    sub.add_parser("crash")
    sub.add_parser("all")
    parser.set_defaults(gui=False)
    args = parser.parse_args(argv)

    handlers = {
        "validate": cmd_validate,
        "receipts": cmd_receipts,
        "plan": cmd_plan,
        "run": cmd_run,
        "invalidate": cmd_invalidate,
        "crash": cmd_crash,
    }
    if args.command == "all":
        for name in ("validate", "receipts", "plan", "run", "invalidate", "crash"):
            handlers[name](args)
            print()
        return 0
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
