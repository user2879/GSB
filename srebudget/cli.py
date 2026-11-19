"""Command-line entry points for generation, runs, injections, and replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .clock import LogicalClock, MonotonicClock
from .engine import DiagnosticEngine
from .errors import SreError
from .loader import load_chain, load_trace
from .models import canonical_json
from .replay import ReplayVerifier
from .runner import BatchRunner
from .samples import sample_trace_records, write_samples
from .storage import SessionStore
from .structured_log import StructuredLogger


DEFAULT_SESSION = "diagnostic-session"


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    data_dir = Path(args.data_dir)
    chain = Path(args.chain) if getattr(args, "chain", None) else data_dir / "chain.json"
    trace = Path(args.trace) if getattr(args, "trace", None) else data_dir / "trace.ndjson"
    db = Path(args.db) if getattr(args, "db", None) else data_dir / "session.sqlite3"
    log = Path(args.log) if getattr(args, "log", None) else data_dir / "logs" / "events.ndjson"
    return chain, trace, db, log


def _open_session(
    store: SessionStore,
    chain: Any,
    trace: Any,
    session_id: str,
    *,
    reset: bool = False,
) -> str:
    try:
        existing = store.get_session(session_id)
    except SreError:
        return store.create_session(chain, trace, session_id=session_id)
    if reset:
        raise SreError(
            "E_SESSION",
            "重置已存在会话需要显式删除数据库或使用新会话标识",
            {"session_id": session_id},
        )
    if (
        existing["chain_digest"] != chain.digest
        or existing["trace_digest"] != trace.content_digest
    ):
        raise SreError(
            "E_SESSION",
            "会话绑定的服务链或调用流与当前输入不一致",
            {
                "session_id": session_id,
                "expected_chain_digest": existing["chain_digest"],
                "actual_chain_digest": chain.digest,
                "expected_trace_digest": existing["trace_digest"],
                "actual_trace_digest": trace.content_digest,
            },
        )
    return session_id


def _command_generate(args: argparse.Namespace) -> int:
    chain_path, trace_path = write_samples(args.data_dir)
    _emit(
        {
            "ok": True,
            "chain": str(chain_path),
            "trace": str(trace_path),
        }
    )
    return 0


def _command_run(args: argparse.Namespace) -> int:
    chain_path, trace_path, db_path, log_path = _paths(args)
    logger = StructuredLogger(log_path)
    chain = load_chain(chain_path)
    trace = load_trace(trace_path, chain)
    clock = LogicalClock(0) if args.clock == "trace" else MonotonicClock()
    with SessionStore(db_path) as store:
        session_id = _open_session(store, chain, trace, args.session_id)
        runner = BatchRunner(
            store,
            session_id,
            chain,
            trace,
            clock=clock,
            follow_trace_time=args.clock == "trace",
        )
        snapshots = []
        processed_calls = 0
        collected_decisions = []
        if args.snapshot_at is not None:
            target = args.snapshot_at
            if target < 0 or target > len(trace):
                raise SreError(
                    "E_SESSION",
                    "snapshot-at 超出调用流范围",
                    {"snapshot_at": target, "total_calls": len(trace)},
                )
            while runner.engine.cursor < len(trace):
                current = runner.engine.cursor
                if current == target:
                    snapshots.append(runner.create_snapshot())
                remaining_to_target = target - current
                size = args.batch_size
                if remaining_to_target > 0:
                    size = min(size, remaining_to_target)
                outcome = runner.run(max_batches=1, batch_size=size)
                if outcome.processed_calls == 0:
                    break
                processed_calls += outcome.processed_calls
                collected_decisions.extend(outcome.decisions)
            if runner.engine.cursor == target:
                snapshots.append(runner.create_snapshot())
        else:
            outcome = runner.run(batch_size=args.batch_size)
            processed_calls = outcome.processed_calls
            collected_decisions.extend(outcome.decisions)
        if args.snapshot:
            snapshots.append(runner.create_snapshot())
        snapshot = snapshots[-1] if snapshots else None
        report = {
            "session_id": session_id,
            "recovery": runner.recovery.as_dict(),
            "processed_calls": processed_calls,
            "final_cursor": runner.engine.cursor,
        }
        if args.show_decisions:
            report["decisions"] = [item.as_dict() for item in collected_decisions]
        report["chain"] = {
            "version": chain.version,
            "digest": chain.digest,
            "services": [service.id for service in chain.services],
        }
        report["services"] = runner.engine.service_rows(clock.now_ms())
        report["snapshot"] = snapshot.as_dict() if snapshot else None
        report["snapshots"] = [item.as_dict() for item in snapshots]
        logger.write(
            "run_completed",
            details={
                "session_id": session_id,
                "processed_calls": processed_calls,
                "final_cursor": runner.engine.cursor,
                "chain_version": chain.version,
            },
        )
    _emit({"ok": True, **report})
    return 0


def _write_injected_chain(data_dir: Path) -> tuple[Path, Any]:
    source = data_dir / "chain.json"
    target = data_dir / "injected" / "chain_corrupt.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8") if source.exists() else "{\n"
    target.write_text(text[: max(8, len(text) // 2)] + "\n", encoding="utf-8")
    return target, load_chain(target)


def _write_injected_trace(
    data_dir: Path, chain: Any
) -> tuple[Path, Any]:
    trace_path = data_dir / "trace.ndjson"
    if trace_path.exists():
        records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    else:
        records = sample_trace_records()
    target = data_dir / "injected" / "trace_timestamp_regression.ndjson"
    target.parent.mkdir(parents=True, exist_ok=True)
    if len(records) >= 2:
        records[1]["timestamp_ms"] = max(0, int(records[0]["timestamp_ms"]) - 1)
    target.write_text(
        "".join(canonical_json(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return target, load_trace(target, chain)


def _write_injected_snapshot(data_dir: Path, chain: Any, trace: Any) -> Path:
    target = data_dir / "injected" / "snapshot_digest_mismatch.sqlite3"
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(target) + suffix)
        if candidate.exists():
            candidate.unlink()
    with SessionStore(target) as store:
        session_id = store.create_session(chain, trace, session_id="inject-snapshot")
        engine = DiagnosticEngine(chain, LogicalClock(0))
        for call in trace.calls[: min(6, len(trace))]:
            engine.clock.follow(call.timestamp_ms)
            engine.process_call(call)
        snapshot = store.create_snapshot(session_id, engine, chain, trace)
        store.corrupt_snapshot_digest(snapshot.snapshot_id)
        store.load_snapshot(snapshot.snapshot_id)
    return target


def _command_inject(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    chain_path, trace_path, db_path, log_path = _paths(args)
    logger = StructuredLogger(log_path)
    details: dict[str, Any] = {}
    try:
        if args.kind == "chain_corrupt":
            target = data_dir / "injected" / "chain_corrupt.json"
            details = {"injected_file": str(target)}
            _write_injected_chain(data_dir)
        elif args.kind == "timestamp_regression":
            chain = load_chain(chain_path)
            target = data_dir / "injected" / "trace_timestamp_regression.ndjson"
            details = {"injected_file": str(target)}
            _write_injected_trace(data_dir, chain)
        else:
            chain = load_chain(chain_path)
            trace = load_trace(trace_path, chain)
            target = data_dir / "injected" / "snapshot_digest_mismatch.sqlite3"
            details = {"injected_db": str(target)}
            _write_injected_snapshot(data_dir, chain, trace)
        raise RuntimeError("injection did not trigger the expected validation error")
    except SreError as error:
        event = logger.error(f"injection_{args.kind}", error)
        _emit(
            {
                "ok": False,
                "injection": args.kind,
                **details,
                "reported": error.as_dict(),
                "log_event": event,
            }
        )
        return {"E_CHAIN": 20, "E_TRACE": 21, "E_SNAPSHOT": 22}.get(error.code, 1)


def _command_replay(args: argparse.Namespace) -> int:
    chain_path, trace_path, db_path, log_path = _paths(args)
    logger = StructuredLogger(log_path)
    chain = load_chain(chain_path)
    trace = load_trace(trace_path, chain)
    with SessionStore(db_path) as store:
        snapshots = store.list_snapshots(args.session_id)
        if not snapshots:
            raise SreError(
                "E_SNAPSHOT",
                "会话没有可重放快照",
                {"session_id": args.session_id},
            )
        snapshot_id = args.snapshot_id or snapshots[-1]["snapshot_id"]
        snapshot = store.load_snapshot(snapshot_id)
        report = ReplayVerifier(chain, trace, snapshot).run()
        logger.write("replay_verified", details=report.as_dict())
    _emit({"ok": True, **report.as_dict()})
    return 0


def _command_ui(args: argparse.Namespace) -> int:
    chain_path, trace_path, db_path, _ = _paths(args)
    try:
        from .ui import launch_ui
    except ImportError as exc:
        raise SreError(
            "E_UI",
            "当前 Python 未安装 Tkinter；Windows CPython 安装时请勾选 Tcl/Tk",
            {"reason": str(exc)},
        ) from exc
    launch_ui(chain_path, trace_path, db_path, args.session_id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srebudget",
        description="服务调用链错误预算与熔断联动诊断工作台",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="生成确定性样本")
    generate.add_argument("--data-dir", default="data")
    generate.set_defaults(func=_command_generate)

    run = subparsers.add_parser("run", help="按批次推进并持久化")
    run.add_argument("--data-dir", default="data")
    run.add_argument("--chain")
    run.add_argument("--trace")
    run.add_argument("--db")
    run.add_argument("--log")
    run.add_argument("--session-id", default=DEFAULT_SESSION)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--clock", choices=("monotonic", "trace"), default="monotonic")
    run.add_argument("--snapshot", action="store_true")
    run.add_argument("--snapshot-at", type=int)
    run.add_argument("--show-decisions", action="store_true")
    run.set_defaults(func=_command_run)

    inject = subparsers.add_parser("inject", help="注入并报告三类业务异常")
    inject.add_argument(
        "kind",
        choices=("chain_corrupt", "timestamp_regression", "snapshot_digest_mismatch"),
    )
    inject.add_argument("--data-dir", default="data")
    inject.add_argument("--chain")
    inject.add_argument("--trace")
    inject.add_argument("--db")
    inject.add_argument("--log")
    inject.set_defaults(func=_command_inject)

    replay = subparsers.add_parser("replay", help="从快照恢复并逐调用比较")
    replay.add_argument("--data-dir", default="data")
    replay.add_argument("--chain")
    replay.add_argument("--trace")
    replay.add_argument("--db")
    replay.add_argument("--log")
    replay.add_argument("--session-id", default=DEFAULT_SESSION)
    replay.add_argument("--snapshot-id")
    replay.set_defaults(func=_command_replay)

    ui = subparsers.add_parser("ui", help="打开 Tkinter 诊断界面")
    ui.add_argument("--data-dir", default="data")
    ui.add_argument("--chain")
    ui.add_argument("--trace")
    ui.add_argument("--db")
    ui.add_argument("--log")
    ui.add_argument("--session-id", default=DEFAULT_SESSION)
    ui.set_defaults(func=_command_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SreError as error:
        data_dir = Path(getattr(args, "data_dir", "data"))
        log_path = Path(getattr(args, "log", None) or data_dir / "logs" / "events.ndjson")
        StructuredLogger(log_path).error("cli_error", error)
        _emit(error.as_dict())
        return {
            "E_CHAIN": 20,
            "E_TRACE": 21,
            "E_SNAPSHOT": 22,
            "E_REPLAY": 23,
        }.get(error.code, 1)


if __name__ == "__main__":
    sys.exit(main())
