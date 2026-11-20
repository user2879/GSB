"""命令行入口：生成 / 运行 / 一致性 / 恢复 / 快照 / 异常注入。"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import ALL_PATHS
from .errors import ArbiterError
from .generator import (
    generate_dataset,
    inject_ledger_digest,
    inject_missing_priority_tenant,
    inject_order_regression,
)
from .ledger import Ledger
from .loader import load_requests, load_spec
from .paths import canonical_requests, run_path
from .runner import (
    choose_and_describe,
    persist_run,
    recover_and_rerun,
    run_all_paths,
    snapshot_and_replay,
    spec_digest,
)
from .snapshot import restore_snapshot
from .util import StructuredLogger


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def _default_paths(workdir):
    return {
        "spec": os.path.join(workdir, "spec.json"),
        "requests": os.path.join(workdir, "requests.ndjson"),
        "ledger": os.path.join(workdir, "ledger.sqlite3"),
        "snapshot": os.path.join(workdir, "snapshot.json"),
        "log": os.path.join(workdir, "events.jsonl"),
    }


def cmd_generate(args):
    info = generate_dataset(
        args.workdir,
        tenant_count=args.tenants,
        pool_count=args.pools,
        request_count=args.requests,
        seed=args.seed,
    )
    _print(info)


def _load_inputs(paths, *, check_global_order=False):
    tenants, pools = load_spec(paths["spec"])
    requests = load_requests(
        paths["requests"], tenants, pools, check_global_order=check_global_order
    )
    return tenants, pools, requests


def cmd_run(args):
    paths = _default_paths(args.workdir)
    logger = StructuredLogger(paths["log"])
    tenants, pools, requests = _load_inputs(paths)
    selection = choose_and_describe(tenants, pools, requests)
    digest = spec_digest(tenants, pools, requests)

    if os.path.exists(paths["ledger"]):
        os.remove(paths["ledger"])
    ledger = Ledger(paths["ledger"])
    ledger.initialize_session(tenants, pools, selection["selected"], digest)
    summary = persist_run(
        tenants, pools, requests, selection["selected"], ledger, logger=logger
    )
    ledger.verify_digest_chain()
    ledger.close()

    result = run_path(selection["selected"], tenants, pools, requests)
    snapshot_and_replay(
        tenants, pools, requests, selection, result, paths["snapshot"]
    )
    logger.info("run_complete", path=selection["selected"], **summary)
    _print(
        {
            "selection": selection,
            "persistence": summary,
            "verdict_counts": _verdict_counts(result),
            "snapshot": paths["snapshot"],
            "ledger": paths["ledger"],
            "replay_verified": True,
        }
    )


def _verdict_counts(result):
    counts = {"GRANT": 0, "QUEUE": 0, "REJECT": 0}
    for outcome in result.outcomes:
        counts[outcome.verdict] += 1
    return counts


def cmd_consistency(args):
    paths = _default_paths(args.workdir)
    tenants, pools, requests = _load_inputs(paths)
    results, report = run_all_paths(tenants, pools, requests)
    report["batch_stats"] = {
        name: {
            "batch_count": len(results[name].batches),
            "total_skipped": sum(b.get("skipped_count", 0) for b in results[name].batches),
            "first_batches": results[name].batches[:3],
        }
        for name in ALL_PATHS
    }
    report["verdict_counts"] = {name: _verdict_counts(results[name]) for name in ALL_PATHS}
    _print(report)
    if not report["identical"]:
        return 2
    return 0


def cmd_crash(args):
    paths = _default_paths(args.workdir)
    logger = StructuredLogger(paths["log"])
    tenants, pools, requests = _load_inputs(paths)
    selection = choose_and_describe(tenants, pools, requests)
    digest = spec_digest(tenants, pools, requests)
    if os.path.exists(paths["ledger"]):
        os.remove(paths["ledger"])
    ledger = Ledger(paths["ledger"])
    ledger.initialize_session(tenants, pools, selection["selected"], digest)
    crashed = False
    try:
        persist_run(
            tenants,
            pools,
            requests,
            selection["selected"],
            ledger,
            crash_after_batch=args.crash_after_batch,
            logger=logger,
        )
    except ArbiterError as exc:
        crashed = True
        logger.error("forced_termination", **exc.to_dict())
        ledger.close()
    if not crashed:
        ledger.close()
    recovery = recover_and_rerun(
        tenants, pools, requests, Ledger(paths["ledger"]), digest,
        selection["selected"], logger=logger,
    )
    logger.info("recovery_complete", **{k: v for k, v in recovery.items() if k != "tenant_state" and k != "pool_state"})
    _print(
        {
            "forced_termination_injected": crashed,
            "recovery": {
                "last_committed_rid": recovery["last_committed_rid"],
                "committed_count": recovery["committed_count"],
                "unknown_count": recovery["unknown_count"],
                "unknown_rids": recovery["unknown_rids"],
                "rerun_ledger": recovery["rerun_ledger"],
            },
            "note": "强制终止时不存在退出报告；仅依据完整提交批次恢复，未提交请求标记未知并重跑。",
        }
    )


def cmd_snapshot_restore(args):
    paths = _default_paths(args.workdir)
    tenants, pools, requests = _load_inputs(paths)
    payload, mismatch, backup = restore_snapshot(paths["snapshot"], tenants, pools)
    replay = run_path(payload["default_path"], tenants, pools, requests)
    from .snapshot import verify_replay

    verify_replay(payload, replay, payload["processed_cursor"])
    _print(
        {
            "format_version": payload["format_version"],
            "mismatch_replaced": mismatch,
            "readonly_backup": backup,
            "default_path": payload["default_path"],
            "replay_verified": True,
            "cursor": payload["processed_cursor"],
        }
    )


def cmd_inject(args):
    paths = _default_paths(args.workdir)
    logger = StructuredLogger(paths["log"])
    kind = args.kind

    if kind == "order":
        dst = paths["requests"] + ".fault"
        meta = inject_order_regression(paths["requests"], dst)
        try:
            tenants, pools = load_spec(paths["spec"])
            load_requests(dst, tenants, pools, check_global_order=True)
        except ArbiterError as exc:
            logger.error("fault_injected", **exc.to_dict(), injection=kind)
            _print({"injection": kind, "error": exc.to_dict(), "meta": meta})
            return 0
        _print({"injection": kind, "status": "未触发异常"})
        return 3

    if kind == "priority":
        dst = paths["spec"] + ".fault"
        meta = inject_missing_priority_tenant(paths["spec"], dst)
        try:
            load_spec(dst)
        except ArbiterError as exc:
            logger.error("fault_injected", **exc.to_dict(), injection=kind)
            _print({"injection": kind, "error": exc.to_dict(), "meta": meta})
            return 0
        return 3

    if kind == "ledger":
        tenants, pools, requests = _load_inputs(paths)
        ledger = Ledger(paths["ledger"])
        ordered = canonical_requests(requests)
        target_rid = ordered[args.ledger_row].rid
        # 选择一条非 REJECT 结局，确保篡改确实改变账本内容
        row = ledger.conn.execute(
            "SELECT rid FROM outcomes WHERE verdict != 'REJECT' ORDER BY rid LIMIT 1"
        ).fetchone()
        if row is not None:
            target_rid = row["rid"]
        meta = inject_ledger_digest(ledger, target_rid)
        ledger.close()
        try:
            damaged = Ledger(paths["ledger"])
            damaged.verify_digest_chain()
        except ArbiterError as exc:
            logger.error("fault_injected", **exc.to_dict(), injection=kind)
            _print({"injection": kind, "error": exc.to_dict(), "meta": meta})
            return 0
        return 3

    raise SystemExit(f"未知注入类型: {kind}")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="quota-arbiter", description="数据库租户资源配额仲裁工作台"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="生成确定性样本")
    gen.add_argument("--workdir", default="data")
    gen.add_argument("--tenants", type=int, default=8)
    gen.add_argument("--pools", type=int, default=3)
    gen.add_argument("--requests", type=int, default=500)
    gen.add_argument("--seed", type=int, default=20261121)
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="按默认路径仲裁并落账、写快照、重放校验")
    run.add_argument("--workdir", default="data")
    run.set_defaults(func=cmd_run)

    cons = sub.add_parser("consistency", help="三路径逐字节一致性校验")
    cons.add_argument("--workdir", default="data")
    cons.set_defaults(func=cmd_consistency)

    crash = sub.add_parser("crash", help="注入强制终止并恢复")
    crash.add_argument("--workdir", default="data")
    crash.add_argument("--crash-after-batch", type=int, default=2)
    crash.set_defaults(func=cmd_crash)

    snap = sub.add_parser("restore", help="打开快照整体替换恢复并重放")
    snap.add_argument("--workdir", default="data")
    snap.set_defaults(func=cmd_snapshot_restore)

    inject = sub.add_parser("inject", help="异常注入 order/priority/ledger")
    inject.add_argument("--workdir", default="data")
    inject.add_argument("kind", choices=["order", "priority", "ledger"])
    inject.add_argument("--ledger-row", type=int, default=0)
    inject.set_defaults(func=cmd_inject)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except ArbiterError as exc:
        payload = exc.to_dict()
        StructuredLogger(os.path.join(getattr(args, "workdir", "data"), "events.jsonl")).error(
            "business_error", **payload
        )
        _print(payload)
        return 1


if __name__ == "__main__":
    sys.exit(main())
