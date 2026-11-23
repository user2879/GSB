from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .compare import compare_incremental_and_full
from .engine import ConfigEngine
from .errors import DiagnosticError
from .injections import run_injection
from .recovery import recover_and_replay
from .structured_log import JsonlLogger
from .subscriptions import SubscriptionSimulator
from .validation import load_bundle
from .graph import topological_order


def _json_print(value: Any) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)
    try:
        sys.stdout.write(text + "\n")
        sys.stdout.flush()
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="config-diag",
        description="Windows 本地配置中心热更新中间件诊断台",
    )
    parser.add_argument("--workspace", default=".", help="可写本地工作区，默认当前目录")
    parser.add_argument("--samples", default=None, help="样本目录，默认 <workspace>/samples")
    parser.add_argument("--db", default=None, help="SQLite 账本路径，默认 <workspace>/runtime/ledger.sqlite3")
    parser.add_argument("--log", default=None, help="JSONL 日志路径，默认 <workspace>/logs/diagnostics.jsonl")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="初始装载样本")
    init.add_argument("--reset", action="store_true", help="保留只读备份后重建账本")

    sub.add_parser("validate", help="强校验三类本地样本并输出稳定拓扑序")
    sub.add_parser("list", help="展示账本、当前视图、快照与依赖状态")

    apply_parser = sub.add_parser("apply", help="应用单键或键组增量变更")
    apply_selector = apply_parser.add_mutually_exclusive_group(required=True)
    apply_selector.add_argument("--sequence", type=int, help="变更序列号")
    apply_selector.add_argument("--all", action="store_true", help="应用全部尚未提交的变更")
    apply_parser.add_argument(
        "--crash-point",
        choices=["after-intent", "before-commit", "after-commit"],
        help="演示强制终止；进程直接退出且不生成退出报告",
    )

    expire = sub.add_parser("expire", help="按单调时钟回收过期键并失效下游")
    expire.add_argument("--wait-ms", type=int, default=0, help="扫描前真实等待毫秒数")

    poll = sub.add_parser("poll", help="使用固定整数种子模拟订阅轮询")
    poll.add_argument("--rounds", type=int, default=3)

    compare = sub.add_parser("compare", help="增量路径与全量重建路径逐键校验")
    compare.add_argument("--inject-diverge", action="store_true", help="示范注入首个 E_DIVERGE")
    compare.add_argument("--now-ns", type=int, default=None, help="测试用冻结单调时钟")

    sub.add_parser("recover", help="恢复后逐键重放校验")

    inject = sub.add_parser("inject", help="注入声明损坏、派生成环或账本摘要不符")
    inject.add_argument("--kind", required=True, choices=["spec", "cycle", "snapshot"])

    sub.add_parser("gui", help="启动 Tkinter 诊断界面")
    return parser


def _make_logger(workspace: Path, path: str | None) -> JsonlLogger:
    return JsonlLogger(Path(path) if path else workspace / "logs" / "diagnostics.jsonl")


def _samples_path(workspace: Path, value: str | None) -> Path:
    return Path(value).resolve() if value else workspace / "samples"


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = _build_parser().parse_args(argv)
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    samples = _samples_path(workspace, args.samples)
    logger = _make_logger(workspace, args.log)

    try:
        if args.command == "validate":
            spec, clients, changes = load_bundle(samples)
            order = topological_order(spec.refs_by_key)
            result = {
                "ok": True,
                "code": "VALID",
                "keys": len(spec.keys),
                "clients": len(clients.clients),
                "operations": len(changes.operations),
                "stable_topology": order,
                "derivation_version": spec.derivation_version,
            }
            logger.emit("INFO_VALIDATE", "samples_validated", message="本地样本强校验通过", keys=len(spec.keys))
            _json_print(result)
            return 0

        if args.command == "compare":
            result = compare_incremental_and_full(
                samples,
                workspace,
                inject_diverge=args.inject_diverge,
                now_ns=args.now_ns,
            )
            logger.emit("INFO_COMPARE", "paths_compared", message="增量与全量路径逐键一致", **result)
            _json_print(result)
            return 0

        if args.command == "inject":
            result = run_injection(args.kind, samples, workspace)
            _json_print(result)
            return 0

        if args.command == "gui":
            from .gui import run_gui

            return run_gui(workspace, samples, db_path=args.db, log_path=args.log)

        engine = ConfigEngine(
            workspace,
            db_path=args.db,
            logger=logger,
        )
        try:
            if args.command == "init":
                spec, clients, _ = load_bundle(samples)
                result = engine.initialize(spec, clients, reset=args.reset)
                _json_print(result)
                return 0

            if args.command == "list":
                _json_print(engine.list_state())
                return 0

            if args.command == "apply":
                _, _, changes = load_bundle(samples)
                if args.all:
                    committed_ops = {
                        item["operation_id"]
                        for item in engine.ledger.journal_rows("committed")
                    }
                    cursor = engine.ledger.max_committed_source_sequence()
                    ops = [
                        item for item in changes.operations
                        if item.sequence > cursor and item.operation_id not in committed_ops
                    ]
                    if not ops:
                        _json_print({"ok": True, "applied": [], "message": "没有待提交变更"})
                        return 0
                    results = [engine.apply_operation(ops[0], crash_point=args.crash_point)]
                    for operation in ops[1:]:
                        results.append(engine.apply_operation(operation))
                else:
                    results = engine.apply_sequence(
                        changes,
                        sequence=args.sequence,
                        crash_point=args.crash_point,
                    )
                _json_print({"ok": True, "applied": results, "open_events": engine.open_events})
                return 0

            if args.command == "expire":
                if args.wait_ms < 0:
                    raise DiagnosticError("E_STATE", "wait-ms 不得为负", field="wait_ms")
                if args.wait_ms:
                    time.sleep(args.wait_ms / 1000.0)
                result = engine.collect_expired()
                result["open_events"] = engine.open_events
                _json_print(result)
                return 0

            if args.command == "poll":
                _, clients, _ = engine._initial_models()
                simulator = SubscriptionSimulator(engine, clients)
                result = simulator.poll(args.rounds)
                _json_print(result)
                return 0

            if args.command == "recover":
                result = recover_and_replay(engine, workspace, logger=logger)
                result["open_events"] = engine.open_events
                _json_print(result)
                return 2 if result.get("needs_rerun") else 0

            raise DiagnosticError("E_STATE", "未知命令", field="command")
        finally:
            engine.close()
    except DiagnosticError as exc:
        logger.emit(
            exc.code,
            "command_failed",
            severity="ERROR",
            message=exc.detail,
            key=exc.key,
            field=exc.field,
            diagnostic_context=exc.context,
        )
        _json_print(exc.to_dict())
        return 1
    except Exception as exc:
        logger.emit(
            "E_INTERNAL",
            "unhandled_exception",
            severity="ERROR",
            message=str(exc),
            traceback=traceback.format_exc(),
        )
        _json_print(
            {
                "ok": False,
                "code": "E_INTERNAL",
                "detail": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
