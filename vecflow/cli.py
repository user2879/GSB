"""Command-line interface for VecFlow."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

from .canonical import canonical_json_text
from .errors import AppError, as_app_error
from .ledger import ExecutionLedger, SnapshotStore
from .logging_utils import JsonLogger
from .model import Workspace
from .planner import compile_workspace
from .service import ExecutionService, load_injected_workspace


DEFAULT_DATA = "samples/data_blocks.json"
DEFAULT_TASKS = "samples/tasks.json"
DEFAULT_LEDGER = "samples/ledger.sqlite"
DEFAULT_SNAPSHOTS = "samples/snapshots"
DEFAULT_LOGS = "logs"


def _add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default=DEFAULT_DATA, help="列式数据块 JSON 路径")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="任务与预算 JSON 路径")


def _add_injection(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--inject", choices=["none", "range", "nullmap", "budget"], default="none")
    group.add_argument("--inject-range", dest="inject", action="store_const", const="range")
    group.add_argument("--inject-nullmap", dest="inject", action="store_const", const="nullmap")
    group.add_argument("--inject-budget", dest="inject", action="store_const", const="budget")


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ledger", default=DEFAULT_LEDGER, help="SQLite 执行账本路径")
    parser.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOTS, help="会话快照目录")
    parser.add_argument("--log-dir", default=DEFAULT_LOGS, help="结构化 JSON 日志目录")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vecflow",
        description="确定性向量化列式执行、预算调度、账本恢复与 Tkinter 工作台",
    )
    parser.add_argument("--version", action="version", version="vecflow 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight", help="检查 CLI、日志目录、NumPy、Tk 与 SQLite")
    preflight.add_argument("--log-dir", default=DEFAULT_LOGS)

    validate = sub.add_parser("validate", help="加载、强校验并编译计划")
    _add_inputs(validate)
    _add_injection(validate)
    validate.add_argument("--log-dir", default=DEFAULT_LOGS)

    plan = sub.add_parser("plan", help="输出规范表达式与确定性分片计划")
    _add_inputs(plan)
    _add_injection(plan)
    plan.add_argument("--log-dir", default=DEFAULT_LOGS)

    run = sub.add_parser("run", help="在全局预算与任务配额下执行所有分片")
    _add_inputs(run)
    _add_injection(run)
    _add_runtime(run)
    run.add_argument("--reset-state", action="store_true", help="演示前清空账本与会话快照")
    run.add_argument("--force-crash-after", type=int, default=None, metavar="N", help="提交 N 个分片后强制终止（恢复演示）")
    run.add_argument("--no-snapshot", action="store_true", help="本次不写会话快照")

    recover = sub.add_parser("recover", help="只依据账本恢复；未提交分片标记 unknown")
    recover.add_argument("--ledger", default=DEFAULT_LEDGER)
    recover.add_argument("--log-dir", default=DEFAULT_LOGS)

    replay = sub.add_parser("replay", help="按账本参数版本重新执行并逐分片比较")
    replay.add_argument("--ledger", default=DEFAULT_LEDGER)
    replay.add_argument("--batch-id", default=None)
    replay.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOTS)
    replay.add_argument("--log-dir", default=DEFAULT_LOGS)

    gui = sub.add_parser("gui", help="启动 Tkinter 编辑与执行工作台")
    _add_inputs(gui)
    _add_runtime(gui)
    gui.add_argument("--smoke", action="store_true", help="构造并短暂刷新界面后退出")
    gui.add_argument("--smoke-ms", type=int, default=800)

    samples = sub.add_parser("generate-samples", help="重新生成本地确定性样本")
    samples.add_argument("--log-dir", default=DEFAULT_LOGS)

    restore = sub.add_parser("restore-snapshot", help="按快照整体替换工作区并报告版本差异")
    _add_inputs(restore)
    restore.add_argument("--snapshot", default=str(Path(DEFAULT_SNAPSHOTS) / "session.json"))
    _add_runtime(restore)

    return parser


def _logger(args: argparse.Namespace) -> JsonLogger:
    return JsonLogger(getattr(args, "log_dir", DEFAULT_LOGS), echo=False)


def _print(payload: Any) -> None:
    print(canonical_json_text(payload, pretty=True))


def _workspace_from_args(args: argparse.Namespace) -> tuple[Workspace, Any]:
    try:
        return load_injected_workspace(args.data, args.tasks, injection=args.inject)
    except AppError:
        raise


def _reset_runtime(args: argparse.Namespace, ledger: ExecutionLedger) -> dict[str, Any]:
    ledger.reset()
    snapshot_path = Path(args.snapshot_dir) / "session.json"
    removed = False
    try:
        snapshot_path.unlink()
        removed = True
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise AppError("E_DATA", f"无法清空快照: {snapshot_path}: {exc}") from exc
    return {"ledger_reset": True, "snapshot_removed": removed, "snapshot_path": str(snapshot_path)}


def _preflight(log_dir: str) -> dict[str, Any]:
    numpy_available = False
    numpy_version = None
    try:
        import numpy

        numpy_available = True
        numpy_version = numpy.__version__
    except Exception:
        pass
    tk_available = importlib.util.find_spec("tkinter") is not None
    display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or ("Windows" if os.name == "nt" else None)
    sqlite_ok = False
    sqlite_error = None
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE probe(value INTEGER)")
        conn.execute("INSERT INTO probe VALUES(1)")
        conn.commit()
        conn.close()
        sqlite_ok = True
    except Exception as exc:
        sqlite_error = str(exc)
    return {
        "ok": numpy_available and sqlite_ok,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cli_parse": True,
        "log_dir": str(Path(log_dir).resolve()),
        "log_dir_writable": True,
        "numpy_available": numpy_available,
        "numpy_version": numpy_version,
        "tkinter_available": tk_available,
        "gui_ready": tk_available and bool(display),
        "display": display,
        "sqlite_available": sqlite_ok,
        "sqlite_error": sqlite_error,
    }


def _command_validate(args: argparse.Namespace, logger: JsonLogger) -> dict[str, Any]:
    workspace, injection = _workspace_from_args(args)
    plan = compile_workspace(workspace.state)
    payload = {
        "ok": True,
        "command": "validate",
        "injection": injection.to_dict(),
        "parameter_version": plan.parameter_version,
        "content_digest": plan.content_digest,
        "plan_digest": plan.plan_digest,
        "block_count": len(workspace.state.blocks.blocks),
        "task_count": len(plan.tasks),
        "shard_count": plan.shard_count,
        "global_budget": plan.global_budget,
        "threads": plan.threads,
    }
    logger.log("validation_ok", **payload)
    return payload


def _command_plan(args: argparse.Namespace, logger: JsonLogger) -> dict[str, Any]:
    workspace, injection = _workspace_from_args(args)
    plan = compile_workspace(workspace.state)
    payload = {"ok": True, "injection": injection.to_dict(), "plan": plan.to_dict()}
    logger.log("plan_compiled", plan_digest=plan.plan_digest, shard_count=plan.shard_count)
    return payload


def _command_run(args: argparse.Namespace, logger: JsonLogger) -> dict[str, Any]:
    ledger = ExecutionLedger(args.ledger)
    reset_info = None
    try:
        if args.reset_state:
            reset_info = _reset_runtime(args, ledger)
            logger.log("runtime_reset", **reset_info)
        workspace, injection = _workspace_from_args(args)
        service = ExecutionService(
            workspace,
            ledger,
            snapshot_store=SnapshotStore(args.snapshot_dir),
            logger=logger,
        )
        report = service.execute(force_crash_after=args.force_crash_after, save_snapshot=not args.no_snapshot)
        latest = ledger.latest_batch(recover=False)
        payload = {
            "ok": report.status == "committed" and not report.stale,
            "command": "run",
            "injection": injection.to_dict(),
            "reset": reset_info,
            "report": report.to_dict(),
            "result_bytes_sha256": report.overall_digest,
            "ledger": latest.to_dict() if latest is not None else None,
        }
        return payload
    finally:
        ledger.close()


def _command_recover(args: argparse.Namespace, logger: JsonLogger) -> dict[str, Any]:
    ledger = ExecutionLedger(args.ledger)
    try:
        interrupted = ledger.recover_incomplete()
        latest = ledger.latest_batch(recover=False)
        payload = {
            "ok": True,
            "command": "recover",
            "interrupted_batches": interrupted,
            "latest": latest.to_dict() if latest is not None else None,
            "exit_report_available": bool(latest.exit_report_allowed) if latest is not None else False,
            "exit_report": latest.result_payload() if latest is not None and latest.exit_report_allowed else None,
            "note": "强制终止时无法生成退出报告；未提交 running/queued 分片已标记 unknown，必须重跑",
        }
        logger.log("recovery_complete", interrupted=interrupted, latest_batch=latest.batch_id if latest else None)
        return payload
    finally:
        ledger.close()


def _placeholder_workspace(ledger: ExecutionLedger) -> Workspace:
    ledger.recover_incomplete()
    latest = ledger.latest_batch(recover=False)
    if latest is None:
        raise AppError("E_REPLAY", "账本为空，没有可重放状态")
    return Workspace.from_export_payload(latest.state_payload())


def _command_replay(args: argparse.Namespace, logger: JsonLogger) -> dict[str, Any]:
    ledger = ExecutionLedger(args.ledger)
    try:
        workspace = _placeholder_workspace(ledger)
        service = ExecutionService(
            workspace,
            ledger,
            snapshot_store=SnapshotStore(args.snapshot_dir),
            logger=logger,
        )
        details = service.replay(batch_id=args.batch_id)
        latest = ledger.latest_batch(recover=False)
        old_batch = ledger.get_batch(details["batch_id"])
        payload = {
            "ok": True,
            "command": "replay",
            "replay": details,
            "source_batch": old_batch.to_dict(),
            "ledger": latest.to_dict() if latest is not None else None,
        }
        return payload
    finally:
        ledger.close()


def _command_restore(args: argparse.Namespace, logger: JsonLogger) -> dict[str, Any]:
    workspace, _ = load_injected_workspace(args.data, args.tasks, injection="none")
    ledger = ExecutionLedger(args.ledger)
    try:
        service = ExecutionService(workspace, ledger, snapshot_store=SnapshotStore(args.snapshot_dir), logger=logger)
        restored = service.restore_snapshot(args.snapshot)
        workspace.persist()
        return {
            "ok": True,
            "command": "restore-snapshot",
            "restore": restored.to_dict(),
            "state": workspace.export_payload(),
            "persisted": True,
        }
    finally:
        ledger.close()


def _command_gui(args: argparse.Namespace, logger: JsonLogger) -> dict[str, Any]:
    from .gui import run_gui

    return run_gui(
        data_path=args.data,
        tasks_path=args.tasks,
        ledger_path=args.ledger,
        snapshot_dir=args.snapshot_dir,
        logger=logger,
        smoke=args.smoke,
        smoke_ms=args.smoke_ms,
    )


def _command_samples(args: argparse.Namespace, logger: JsonLogger) -> dict[str, Any]:
    from .samples import generate_samples

    result = generate_samples(Path.cwd() / "samples")
    logger.log("samples_generated", **result)
    return {"ok": True, "command": "generate-samples", **result}


def dispatch(args: argparse.Namespace, logger: JsonLogger) -> dict[str, Any]:
    handlers = {
        "validate": _command_validate,
        "plan": _command_plan,
        "run": _command_run,
        "recover": _command_recover,
        "replay": _command_replay,
        "restore-snapshot": _command_restore,
        "gui": _command_gui,
        "generate-samples": _command_samples,
    }
    if args.command == "preflight":
        payload = _preflight(args.log_dir)
        logger.log("preflight", **payload)
        return payload
    return handlers[args.command](args, logger)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger: JsonLogger | None = None
    try:
        logger = _logger(args)
        logger.log("cli_start", argv=list(sys.argv[1:] if argv is None else argv), command=args.command)
        payload = dispatch(args, logger)
        if payload.get("ok", True):
            logger.log("cli_ok", command=args.command)
        else:
            logger.log("cli_not_ok", level="WARNING", command=args.command)
        _print(payload)
        return 0 if payload.get("ok", True) else 1
    except AppError as exc:
        if logger is not None:
            try:
                logger.error("cli_error", exc, command=getattr(args, "command", None))
            except AppError:
                pass
        _print(exc.as_dict())
        return exc.exit_code
    except KeyboardInterrupt:
        error = AppError("E_INTERNAL", "用户中断")
        _print(error.as_dict())
        return 130
    except Exception as exc:
        error = as_app_error(exc)
        if logger is not None:
            try:
                logger.error("cli_internal_error", error, command=getattr(args, "command", None))
            except AppError:
                pass
        _print(error.as_dict())
        return error.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
