"""命令行入口。

子命令：
  gen-sample  生成声明与事件样本
  run         启动节点子进程并跑完整下发/回收流程
  crash-demo  运行到指定事件后强制终止节点，再重启恢复、续跑、重放
  verify      独立只读重放校验
  ui          打开 Tkinter 只读水位界面
"""

import argparse
import json
import sys
from pathlib import Path

from .clock import MonoClock
from .engine import Engine
from .errors import DiagError
from .replay import verify
from .samplegen import write_sample
from .spec import load_spec, spec_digest


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def cmd_gen_sample(args):
    path = write_sample(args.spec)
    print(f"样本已生成: {path}")


def _make_engine(args):
    spec = load_spec(args.spec)
    return Engine(spec, args.spec, args.db, clock=MonoClock())


def cmd_run(args):
    engine = _make_engine(args)
    try:
        engine.start_nodes()
        outcome = engine.run()
        _print({"outcome": outcome, "water": engine.water_report()})
        result = verify(args.spec, args.db)
        _print({"verify": result})
    finally:
        engine.stop_nodes()
        engine.close()


def cmd_crash_demo(args):
    """两段式：崩溃前运行到提交边界 -> SIGKILL 节点 -> 新引擎依据账本
    恢复已提交事实、回收未知批次占用 -> 续跑未提交事件 -> 重放校验。"""
    spec = load_spec(args.spec)
    crash_seq = int(args.crash_seq)
    engine = Engine(spec, args.spec, args.db, clock=MonoClock())
    try:
        engine.start_nodes()
        first = engine.run_until(crash_seq)
        _print({"phase": "before_crash", **first})
        engine.force_kill_node(args.crash_node)
    finally:
        engine.stop_nodes()
        engine.close()

    engine2 = Engine(spec, args.spec, args.db, clock=MonoClock())
    try:
        engine2.start_nodes(skip={args.crash_node})
        last_seq, snap = engine2.recover()
        _print({"phase": "recovered", "last_committed_seq": last_seq,
                "unknown_batches": snap["unknown"],
                "active_leases_restored": len(snap["leases"])})
        # 重启被强制终止的节点，并回放该节点已提交租约
        engine2.restart_node(args.crash_node)
        node_leases = [l for l in snap["leases"]
                       if l["node_id"] == args.crash_node]
        if node_leases:
            engine2.nodes[args.crash_node].request(
                {"type": "restore",
                 "leases": [{
                     "lease_id": l["lease_id"], "shard_id": l["shard_id"],
                     "quota_version": l["quota_version"],
                     "grant_ms": l["grant_ms"], "expire_ms": l["expire_ms"],
                     "renew_count": l["renew_count"],
                     "lease_version": l["lease_version"]}
                     for l in node_leases]})
        second = engine2.run(start_after_seq=last_seq)
        _print({"phase": "after_recovery", **second,
                "water": engine2.water_report()})
        result = verify(args.spec, args.db)
        _print({"verify": result})
    finally:
        engine2.stop_nodes()
        engine2.close()


def cmd_verify(args):
    _print(verify(args.spec, args.db))


def cmd_ui(args):
    from .ui import launch_ui
    launch_ui(args.spec, args.db)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="edge-diag", description="边缘缓存分片下发诊断工作台")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gen-sample")
    p.add_argument("--spec", default="samples/demo_spec.json")
    p.set_defaults(func=cmd_gen_sample)

    p = sub.add_parser("run")
    p.add_argument("--spec", default="samples/demo_spec.json")
    p.add_argument("--db", default="state/ledger.sqlite3")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("crash-demo")
    p.add_argument("--spec", default="samples/demo_spec.json")
    p.add_argument("--db", default="state/ledger.sqlite3")
    p.add_argument("--crash-seq", default="5")
    p.add_argument("--crash-node", default="node-b")
    p.set_defaults(func=cmd_crash_demo)

    p = sub.add_parser("verify")
    p.add_argument("--spec", default="samples/demo_spec.json")
    p.add_argument("--db", default="state/ledger.sqlite3")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("ui")
    p.add_argument("--spec", default="samples/demo_spec.json")
    p.add_argument("--db", default="state/ledger.sqlite3")
    p.set_defaults(func=cmd_ui)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except DiagError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
