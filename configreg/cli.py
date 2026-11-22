"""平台命令行入口。

子命令
------
``validate``  仅做强校验并打印清单/注册表摘要
``diff``      输出差异度量、影响链与定点误差界（不启动节点）
``run``       启动节点子进程、分批采集、判定回归并写账本/快照
``replay``    重放采集并与账本逐项核对（不一致抛 E_REPLAY）
"""
from __future__ import annotations

import argparse
import json
from typing import Any

from .collect import SessionParams
from .diff import build_diff_report
from .errors import PlatformError
from .manifest import load_manifest_pair
from .session import Session


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _params_from_args(args: argparse.Namespace) -> SessionParams:
    return SessionParams(
        sample_seed=args.seed,
        precision=args.precision,
        global_concurrency=args.global_concurrency,
        per_node_concurrency=args.per_node_concurrency,
        socket_timeout_ms=args.socket_timeout_ms,
        max_attempts=args.max_attempts,
        retry_backoff_ms=args.retry_backoff_ms,
        batch_size=args.batch_size,
        max_elapsed_delta_ms=args.max_elapsed_delta_ms,
        max_mismatch_nodes=args.max_mismatch_nodes,
        strict_timing=args.strict_timing,
    )


def cmd_validate(args: argparse.Namespace) -> int:
    baseline, candidate, registry = load_manifest_pair(
        args.baseline, args.candidate, args.registry
    )
    _print_json(
        {
            "ok": True,
            "registry_nodes": registry.node_ids(),
            "baseline": {
                "version": baseline.version,
                "strategy": baseline.strategy,
                "items": baseline.keys(),
                "topological_order": baseline.topological_order(),
                "digest": baseline.source_digest,
            },
            "candidate": {
                "version": candidate.version,
                "strategy": candidate.strategy,
                "items": candidate.keys(),
                "topological_order": candidate.topological_order(),
                "digest": candidate.source_digest,
            },
        }
    )
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    baseline, candidate, _ = load_manifest_pair(args.baseline, args.candidate, args.registry)
    report = build_diff_report(baseline, candidate, args.precision)
    _print_json(report.to_dict())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    baseline, candidate, registry = load_manifest_pair(args.baseline, args.candidate, args.registry)
    params = _params_from_args(args)
    session = Session(
        baseline,
        candidate,
        registry,
        params,
        ledger_path=args.ledger,
        log_dir=args.log_dir,
    )
    if args.crash_after_batches is not None:
        session.crash_after_batches = args.crash_after_batches

    def progress(done: int, total: int) -> None:
        if args.quiet:
            return
        print(f"  采集进度 {done}/{total}")

    result = session.run(progress=progress)
    payload = {
        "ok": True,
        "identity": result.identity.to_dict(),
        "replacement": result.replacement,
        "unknown_items_on_resume": result.unknown_items,
        "batches": [
            {"batch_id": record.batch_id, "tasks": len(record.task_keys), "content_digest": record.content_digest}
            for record in result.batches
        ],
        "diff_digest": result.diff_report.report_digest,
        "regression": result.regression_report.to_dict(),
        "snapshot_digest": result.snapshot_digest,
    }
    _print_json(payload)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    baseline, candidate, registry = load_manifest_pair(args.baseline, args.candidate, args.registry)
    params = _params_from_args(args)
    session = Session(baseline, candidate, registry, params, ledger_path=args.ledger, log_dir=args.log_dir)
    result = session.replay_and_verify()
    _print_json(result)
    session.ledger.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="configreg", description="配置分发结果验证与差异回归分析平台")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--baseline", default="data/baseline.json")
        sp.add_argument("--candidate", default="data/candidate.json")
        sp.add_argument("--registry", default="data/registry.json")

    validate_parser = sub.add_parser("validate", help="强校验清单与注册表")
    add_common(validate_parser)
    validate_parser.set_defaults(func=cmd_validate)

    diff_parser = sub.add_parser("diff", help="差异度量、影响链与定点误差界")
    add_common(diff_parser)
    diff_parser.add_argument("--precision", type=int, default=4)
    diff_parser.set_defaults(func=cmd_diff)

    run_parser = sub.add_parser("run", help="完整采集、回归判定与账本写入")
    add_common(run_parser)
    _add_session_args(run_parser)
    run_parser.set_defaults(func=cmd_run)

    replay_parser = sub.add_parser("replay", help="重放并与账本核对")
    add_common(replay_parser)
    _add_session_args(replay_parser)
    replay_parser.set_defaults(func=cmd_replay)
    return parser


def _add_session_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ledger", default="artifacts/ledger.sqlite3")
    parser.add_argument("--log-dir", default="artifacts/node_logs")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--global-concurrency", type=int, default=3)
    parser.add_argument("--per-node-concurrency", type=int, default=2)
    parser.add_argument("--socket-timeout-ms", type=int, default=1200)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-backoff-ms", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-elapsed-delta-ms", type=int, default=50)
    parser.add_argument("--max-mismatch-nodes", type=int, default=0)
    parser.add_argument("--strict-timing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--crash-after-batches",
        type=int,
        default=None,
        help="强制终止演练：完整提交指定数量批次后立即终止，不生成退出报告",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PlatformError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    except KeyboardInterrupt as exc:
        print(
            json.dumps(
                {
                    "code": "E_TERMINATED",
                    "message": str(exc),
                    "note": "强制终止时不生成退出报告；仅已完整提交的批次可用于恢复，未提交项重跑",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
