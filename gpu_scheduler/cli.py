from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

from .app import export_receipts, recover_only, run_scheduler, verify_receipt_export
from .diagnostics import JsonLogger, print_json
from .errors import AppError, SpecError
from .hashing import canonical_text
from .ledger import Ledger
from .manager import occupy_port_until_released
from .snapshot import restore_and_verify
from .spec import load_spec
from .scheduler import SchedulerEngine


DEFAULT_WORKDIR = Path("runtime/default")
INJECT_ALIASES = {
    "port": "port",
    "e_port": "port",
    "cycle": "cycle",
    "e_cycle": "cycle",
    "snapshot": "snapshot",
    "e_snapshot": "snapshot",
}


def injection_kind(value: str | None) -> str | None:
    if value is None:
        return None
    return INJECT_ALIASES.get(value.lower())


class UiUnavailableError(AppError):
    code = "E_UI"
    exit_code = 8


def _logger(workdir: Path) -> JsonLogger:
    workdir.mkdir(parents=True, exist_ok=True)
    return JsonLogger(workdir / "logs" / "cli.jsonl")


def inject_cycle_file(spec_path: str, workdir: Path) -> Path:
    raw = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SpecError("cannot inject cycle without tasks", locator={"field": "tasks"})
    target = tasks[0]
    target["shard_count"] = max(2, int(target.get("shard_count", 2) or 2))
    target["shard_dependencies"] = [
        {"shard": 0, "requires": [1]},
        {"shard": 1, "requires": [0]},
    ]
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "injected-cycle.json"
    path.write_text(canonical_text(raw) + "\n", encoding="utf-8")
    return path


def cmd_validate(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir or DEFAULT_WORKDIR)
    logger = _logger(workdir)
    spec_path = Path(args.spec)
    inject = injection_kind(args.inject)
    if inject == "cycle":
        spec_path = inject_cycle_file(args.spec, workdir)
    spec = load_spec(spec_path)
    result = {
        "ok": True,
        "code": "OK",
        "spec_path": str(spec_path),
        "spec_digest": spec.spec_digest,
        "pools": [p.id for p in spec.pools],
        "tasks": [t.id for t in spec.tasks],
        "topological_order": __import__("gpu_scheduler.graph", fromlist=["DependencyGraph"]).DependencyGraph.build(spec).ensure_acyclic(),
    }
    logger.event("INFO", "spec_validated", spec_digest=spec.spec_digest)
    print_json(result)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir or DEFAULT_WORKDIR)
    logger = _logger(workdir)
    inject = injection_kind(args.inject)
    spec_path = args.spec
    if inject == "cycle":
        spec_path = str(inject_cycle_file(args.spec, workdir))
    spec = load_spec(spec_path)
    occupant = None
    if inject == "port":
        pool = sorted(spec.pools, key=lambda p: p.id)[0]
        occupant = occupy_port_until_released(pool.port)
        logger.event("WARNING", "port_occupation_injected", pool_id=pool.id, port=pool.port)
    try:
        report = run_scheduler(
            spec_path,
            workdir,
            global_gpu_quota=args.global_gpu_quota,
            fail_shards=set(args.fail_shard or []),
            retry_shards=set(args.retry_shard or []),
            crash_ready_file=args.crash_ready_file,
            max_wall_seconds=args.max_wall_seconds,
        )
    finally:
        if occupant is not None:
            occupant.close()

    if inject == "snapshot":
        ledger = Ledger(workdir / "ledger.sqlite3")
        try:
            tampered_round = ledger.tamper_latest_content_digest()
        finally:
            ledger.close()
        logger.event("WARNING", "snapshot_digest_mismatch_injected", round_index=tampered_round)
        restore_and_verify(workdir / "session-snapshot.json", workdir, load_spec(spec_path))

    print_json(report)
    if report.get("signal"):
        return 128 + signal.Signals[report["signal"]].value
    counts = report.get("counts", {})
    if any(counts.get(state, 0) for state in ("failed", "blocked", "unschedulable")):
        return 7
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    result = recover_only(args.spec, args.workdir)
    print_json(result)
    return 0


def cmd_quota(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir or DEFAULT_WORKDIR)
    spec = load_spec(args.spec)
    logger = _logger(workdir)
    engine = SchedulerEngine(spec, workdir, executor_manager=None, logger=logger)
    try:
        result = engine.adjust_quota(args.global_gpu_quota)
        snapshot_path = None
        if args.write_snapshot:
            from .app import write_session_snapshot

            snapshot_path = str(write_session_snapshot(spec, engine))
        print_json({**result, "snapshot_path": snapshot_path, "workdir": str(workdir)})
        return 0
    finally:
        engine.close()


def cmd_snapshot_verify(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    result = restore_and_verify(args.snapshot, args.workdir, spec)
    print_json({"ok": True, **result})
    return 0


def cmd_receipts_export(args: argparse.Namespace) -> int:
    result = export_receipts(args.workdir, args.output)
    print_json({"ok": True, **result})
    return 0


def cmd_receipts_verify(args: argparse.Namespace) -> int:
    affected = set(args.affected or [])
    if args.affected_file:
        payload = json.loads(Path(args.affected_file).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            affected.update(payload.get("affected", []))
        else:
            affected.update(payload)
    result = verify_receipt_export(args.workdir, args.before, affected)
    print_json(result)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        import tkinter  # noqa: F401
    except Exception as exc:
        raise UiUnavailableError(
            "Tkinter is unavailable in this Python installation; use the official Python/Tk Windows package",
            locator={"error": str(exc)},
        ) from exc
    from .ui import launch_ui

    launch_ui(Path(args.spec), Path(args.workdir or DEFAULT_WORKDIR))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .verification import run_acceptance

    result = run_acceptance(Path(args.workdir or "runtime/acceptance"), only=args.only)
    print_json(result)
    return 0 if result.get("ok") else 9


def cmd_doctor(_args: argparse.Namespace) -> int:
    import numpy

    tk = {"available": True}
    try:
        import tkinter

        tk["version"] = str(tkinter.TkVersion)
    except Exception as exc:
        tk = {"available": False, "error": str(exc)}
    result = {
        "ok": True,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "tkinter": tk,
        "loopback_host": "127.0.0.1",
    }
    print_json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gpu_scheduler", description="GPU pool scheduler with replayable SQLite ledger")
    parser.add_argument("--workdir", help="runtime/SQLite/log directory")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate JSON declaration")
    validate.add_argument("spec")
    validate.add_argument("--inject", choices=["cycle", "E_CYCLE"])
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="run scheduler and loopback executors")
    run.add_argument("spec")
    run.add_argument("--global-gpu-quota", type=int)
    run.add_argument("--fail-shard", action="append")
    run.add_argument("--retry-shard", action="append")
    run.add_argument("--crash-ready-file")
    run.add_argument("--max-wall-seconds", type=float, default=60.0)
    run.add_argument("--inject", choices=["port", "cycle", "snapshot", "E_PORT", "E_CYCLE", "E_SNAPSHOT"])
    run.set_defaults(func=cmd_run)

    recover = sub.add_parser("recover", help="recover unknown shards from SQLite ledger without scheduling")
    recover.add_argument("spec")
    recover.set_defaults(func=cmd_recover)

    quota = sub.add_parser("quota", help="append a quota-adjustment watermark round")
    quota.add_argument("spec")
    quota.add_argument("--global-gpu-quota", type=int, required=True)
    quota.add_argument("--write-snapshot", action="store_true")
    quota.set_defaults(func=cmd_quota)

    snapshot = sub.add_parser("snapshot-verify", help="whole-restore snapshot and replay ledger")
    snapshot.add_argument("spec")
    snapshot.add_argument("snapshot")
    snapshot.set_defaults(func=cmd_snapshot_verify)

    receipts = sub.add_parser("receipts-export")
    receipts.add_argument("workdir")
    receipts.add_argument("output")
    receipts.set_defaults(func=cmd_receipts_export)

    verify_receipts = sub.add_parser("receipts-verify")
    verify_receipts.add_argument("workdir")
    verify_receipts.add_argument("--before", required=True)
    verify_receipts.add_argument("--affected", action="append")
    verify_receipts.add_argument("--affected-file")
    verify_receipts.set_defaults(func=cmd_receipts_verify)

    ui = sub.add_parser("ui", help="launch Tkinter diagnostic console")
    ui.add_argument("spec")
    ui.set_defaults(func=cmd_ui)

    verify = sub.add_parser("verify", help="run local acceptance drills")
    verify.add_argument("--workdir", default=argparse.SUPPRESS, help="acceptance output directory")
    verify.add_argument("--only", choices=["watermark", "preemption", "propagation", "kill", "errors", "all"])
    verify.set_defaults(func=cmd_verify)

    sub.add_parser("doctor").set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workdir = Path(getattr(args, "workdir", None) or DEFAULT_WORKDIR)
    logger = _logger(workdir)
    try:
        return int(args.func(args) or 0)
    except AppError as exc:
        record = logger.error(exc)
        print_json({"ok": False, **exc.to_dict(), "log_event": record})
        return exc.exit_code
    except KeyboardInterrupt:
        logger.event("WARNING", "keyboard_interrupt_before_signal_report")
        print_json({"ok": False, "code": "E_INTERRUPT", "message": "interrupted"})
        return 130
    except Exception as exc:
        logger.event("ERROR", "internal_error", error_type=type(exc).__name__, error=str(exc))
        print_json({"ok": False, "code": "E_INTERNAL", "message": str(exc), "type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
