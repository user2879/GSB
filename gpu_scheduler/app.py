from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from typing import Any

from .diagnostics import JsonLogger
from .hashing import canonical_text, digest_obj
from .manager import ExecutorManager
from .scheduler import SchedulerEngine
from .snapshot import build_snapshot_content, write_snapshot
from .spec import Spec, load_spec
from .viewmodel import write_ui_cache


def make_logger(workdir: Path) -> JsonLogger:
    log_path = workdir / "logs" / "app.jsonl"
    return JsonLogger(log_path)


def write_exit_report(engine: SchedulerEngine, signame: str | None, received_wall_ms: int | None) -> Path:
    reports_dir = engine.workdir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    content = {
        "format_version": 1,
        "complete": True,
        "pid": __import__("os").getpid(),
        "signal": signame,
        "received_wall_ms": received_wall_ms,
        "completed_wall_ms": int(time.time() * 1000),
        "ledger_cursor": engine.head_round,
        "ledger_head_digest": engine.ledger.head_chain_digest,
        "state": engine.state_report(),
        "watermarks_at_exit": engine._build_watermarks(),
        "shards": {sid: {k: v for k, v in state.items() if k != "future"} for sid, state in sorted(engine.shards.items())},
    }
    content["report_digest"] = digest_obj({k: v for k, v in content.items() if k != "report_digest"})
    stamp = int(time.time() * 1000)
    signal_part = signame.lower() if signame else "normal"
    path = reports_dir / f"exit-report-{__import__('os').getpid()}-{signal_part}-{stamp}.json"
    path.write_text(canonical_text(content) + "\n", encoding="utf-8")
    latest = reports_dir / "latest-exit-report.json"
    latest.write_text(canonical_text(content) + "\n", encoding="utf-8")
    return path


def write_session_snapshot(spec: Spec, engine: SchedulerEngine) -> Path:
    content = build_snapshot_content(spec, engine.ledger, session_id=engine.workdir.name)
    path = engine.workdir / "session-snapshot.json"
    write_snapshot(path, content)
    top_level = Path("snapshots") / f"{engine.workdir.name}-head.json"
    try:
        write_snapshot(top_level, content)
    except OSError:
        # Workdir portability: the authoritative copy remains beside the ledger.
        pass
    return path


def run_scheduler(
    spec_path: str | Path,
    workdir: str | Path,
    *,
    global_gpu_quota: int | None = None,
    fail_shards: set[str] | None = None,
    retry_shards: set[str] | None = None,
    crash_ready_file: str | Path | None = None,
    max_wall_seconds: float = 60.0,
    install_signals: bool = True,
    write_snapshot_file: bool = True,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(workdir)
    manager = ExecutorManager(spec, workdir, logger)
    engine: SchedulerEngine | None = None
    signal_state: dict[str, Any] = {"signal": None, "received_wall_ms": None}

    def handle_signal(signum: int, _frame: Any) -> None:
        signame = signal.Signals(signum).name
        signal_state["signal"] = signame
        signal_state["received_wall_ms"] = int(time.time() * 1000)
        logger.event("WARNING", "signal_received", signal=signame, disposition="graceful_shard_boundary_stop")
        if engine is not None:
            engine.request_stop()

    previous_handlers: dict[int, Any] = {}
    try:
        manager.start()
        engine = SchedulerEngine(
            spec,
            workdir,
            executor_manager=manager,
            logger=logger,
            global_gpu_quota=global_gpu_quota,
            fail_shards=fail_shards,
            retry_shards=retry_shards,
            max_wall_seconds=max_wall_seconds,
            view_writer=lambda eng: write_ui_cache(eng, workdir / "ui-cache.json"),
        )
        if crash_ready_file:
            engine.crash_ready_path = Path(crash_ready_file)
        if install_signals:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, handle_signal)
        report = engine.run_until_complete()
        exit_report_path: Path | None = None
        if signal_state["signal"]:
            exit_report_path = write_exit_report(engine, signal_state["signal"], signal_state["received_wall_ms"])
            report["exit_report"] = str(exit_report_path)
            report["signal"] = signal_state["signal"]
        if write_snapshot_file:
            report["snapshot_path"] = str(write_session_snapshot(spec, engine))
        report["recovery_report"] = engine.recovery_report
        report["spec_digest"] = spec.spec_digest
        report["workdir"] = str(workdir)
        return report
    finally:
        if install_signals:
            for sig, handler in previous_handlers.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass
        if engine is not None:
            engine.close()
        manager.stop()


def recover_only(spec_path: str | Path, workdir: str | Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    workdir = Path(workdir)
    logger = make_logger(workdir)
    engine = SchedulerEngine(spec, workdir, executor_manager=None, logger=logger)
    try:
        return engine.recovery_report | {"workdir": str(workdir), "head_round": engine.head_round}
    finally:
        engine.close()


def export_receipts(workdir: str | Path, output: str | Path) -> dict[str, Any]:
    from .ledger import Ledger

    ledger = Ledger(Path(workdir) / "ledger.sqlite3")
    try:
        data = ledger.export_receipt_snapshot()
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_text(data) + "\n", encoding="utf-8")
        return {"output": str(path), "shard_count": len(data["receipts"])}
    finally:
        ledger.close()


def verify_receipt_export(workdir: str | Path, before_path: str | Path, affected: set[str]) -> dict[str, Any]:
    from .ledger import Ledger

    before = json.loads(Path(before_path).read_text(encoding="utf-8"))["receipts"]
    ledger = Ledger(Path(workdir) / "ledger.sqlite3")
    unchanged: list[str] = []
    checked_affected: list[str] = []
    try:
        all_receipts = ledger.all_receipts()
        for shard_id, old in sorted(before.items()):
            if shard_id in affected:
                checked_affected.append(shard_id)
                continue
            receipt_id = old["receipt_id"]
            row = ledger.conn.execute("SELECT * FROM receipts WHERE receipt_id=?", (receipt_id,)).fetchone()
            if row is None:
                raise AssertionError(f"original receipt row disappeared for unaffected shard {shard_id}")
            if row["canonical_receipt_json"].encode("utf-8") != old["canonical_receipt_json"].encode("utf-8"):
                raise AssertionError(f"receipt bytes changed for unaffected shard {shard_id}")
            if row["receipt_digest"] != old["receipt_digest"]:
                raise AssertionError(f"receipt digest changed for unaffected shard {shard_id}")
            latest = ledger.latest_receipts().get(shard_id)
            if latest is None or latest["receipt_id"] != receipt_id:
                raise AssertionError(f"unaffected shard {shard_id} acquired a replacement receipt")
            unchanged.append(shard_id)
        return {"ok": True, "byte_identical_shards": unchanged, "skipped_affected": sorted(checked_affected)}
    finally:
        ledger.close()
