"""Command-line runner: deterministic demos, exception injection, crash replay.

Scenarios (executed in order by ``demo``):
  1. contention - capacity waterline, priority queue and stable drain
  2. expiry      - real monotonic lease expiry and reclamation
  3. replay      - forced termination, UNKNOWN marking, ledger recovery replay

Exception injection flags (each writes a structured log line and exits 3):
  --inject-timing       renew_interval_ms >= lease_duration_ms  -> E_TIMING
  --inject-ref          holder target domain missing            -> E_REF
  --inject-snapshot     corrupt snapshot digest                 -> E_SNAPSHOT
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .diagnostics import build_dashboard, render_text, TK_AVAILABLE, TK_IMPORT_ERROR
from .engine import DistLockEngine, EngineCrash
from .errors import DistLockError
from .logging_utils import StructuredLogger
from .snapshot import read_snapshot
from .spec import load_events, load_spec

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
DEFAULT_BASE_PORT = 39000


def reset_workdir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load(scenario: str):
    base = EXAMPLES / scenario
    domains, holders, session = load_spec(base / "spec.json")
    events = load_events(base / "events.ndjson", domains, holders)
    return domains, holders, session, events


# ------------------------------------------------------------- exception inject
def inject_timing(logger: StructuredLogger) -> int:
    base = EXAMPLES / "contention"
    spec = json.loads((base / "spec.json").read_text(encoding="utf-8"))
    spec["domains"][0]["renew_interval_ms"] = spec["domains"][0]["lease_duration_ms"]
    tmp = base / "_inject_timing_spec.json"
    tmp.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    try:
        load_spec(tmp)
    except DistLockError as exc:
        logger.error(exc.code, "spec_validation_rejected", context=exc.context, message=exc.message)
        print(f"injection rejected as expected: {exc}")
        return 0
    finally:
        tmp.unlink(missing_ok=True)
    return 1


def inject_ref(logger: StructuredLogger) -> int:
    base = EXAMPLES / "contention"
    spec = json.loads((base / "spec.json").read_text(encoding="utf-8"))
    spec["holders"][0]["target_domains"].append("locks-does-not-exist")
    tmp = base / "_inject_ref_spec.json"
    tmp.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    try:
        load_spec(tmp)
    except DistLockError as exc:
        logger.error(exc.code, "spec_validation_rejected", context=exc.context, message=exc.message)
        print(f"injection rejected as expected: {exc}")
        return 0
    finally:
        tmp.unlink(missing_ok=True)
    return 1


def inject_snapshot(logger: StructuredLogger, workdir: Path) -> int:
    snap_path = workdir / "snapshots" / "session.snapshot.json"
    if not snap_path.exists():
        print("snapshot missing; run replay scenario first", file=sys.stderr)
        return 1
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    snap["digest"] = "0" * 64  # force content digest mismatch
    snap_path.write_text(json.dumps(snap), encoding="utf-8")
    try:
        read_snapshot(snap_path)
    except DistLockError as exc:
        logger.error(exc.code, "snapshot_validation_rejected", context=exc.context, message=exc.message)
        print(f"injection rejected as expected: {exc}")
        return 0
    return 1


# -------------------------------------------------------------------- scenarios
def run_contention(logger: StructuredLogger, base_port: int) -> dict[str, Any]:
    domains, holders, session, events = load("contention")
    workdir = ROOT / "workdir" / "contention"
    reset_workdir(workdir)
    with DistLockEngine(domains, holders, session, workdir,
                        base_port=base_port) as engine:
        engine.events = events
        summary = engine.run(events)
        view = engine.diagnostic_view()
        logger.log("INFO", "scenario_contention_done", **summary)
    return {"summary": summary, "view": view}


def run_expiry(logger: StructuredLogger, base_port: int) -> dict[str, Any]:
    domains, holders, session, events = load("expiry")
    workdir = ROOT / "workdir" / "expiry"
    reset_workdir(workdir)
    with DistLockEngine(domains, holders, session, workdir,
                        base_port=base_port) as engine:
        engine.events = events
        # Events 1-3 first; then sleep beyond the 400ms lease to force real
        # monotonic expiry inside the arbiter, then process the remainder.
        first = [e for e in events if e.seq <= 3]
        rest = [e for e in events if e.seq > 3]
        engine.run(first)
        time.sleep(0.55)  # > lease_duration_ms(400): real deadline passes
        summary = engine.run(rest)
        view = engine.diagnostic_view()
        verdicts = [r["receipt"]["verdict"] for r in engine.results]
        logger.log("INFO", "scenario_expiry_done", verdicts=verdicts, **summary)
    return {"summary": summary, "view": view, "verdicts": verdicts}


def run_replay(logger: StructuredLogger, base_port: int,
               crash_seq: int = 5) -> dict[str, Any]:
    domains, holders, session, events = load("replay")
    workdir = ROOT / "workdir" / "replay"
    reset_workdir(workdir)

    # First process with a forced crash armed right after inflight marking.
    crashed: EngineCrash | None = None
    engine = DistLockEngine(domains, holders, session, workdir,
                            base_port=base_port,
                            crash_after_inflight_seq=crash_seq)
    engine.events = events
    engine.start()
    try:
        engine.run(events)
    except EngineCrash as exc:
        crashed = exc
        logger.log("ERROR", "forced_termination_simulated",
                   code="E_CRASH", message=str(exc), crash_seq=crash_seq)
    engine.scheduler.shutdown()
    engine.ledger.close()

    # New engine instance over the same durable files: recovery path.
    engine2 = DistLockEngine(domains, holders, session, workdir,
                             base_port=base_port)
    engine2.events = events
    engine2.start()
    try:
        recovery = engine2.recover()
        logger.log("INFO", "recovery_completed", **{
            "cursor": recovery["cursor"],
            "unknown_count": len(recovery["unknown_events"]),
            "replayed": recovery["replayed_event_count"]})
        cont = engine2.run(events)  # resume from ledger cursor
        verify = engine2.verify_snapshot_replay()
        view = engine2.diagnostic_view()
    finally:
        engine2.close()
    return {"crashed": str(crashed), "recovery": recovery,
            "continuation": cont, "verify": verify, "view": view}


# ------------------------------------------------------------------------- demo
def cmd_demo(args: argparse.Namespace) -> int:
    log_path = ROOT / "sample_logs" / "demo.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    logger = StructuredLogger(log_path)
    report: dict[str, Any] = {}
    try:
        print("== Scenario 1: contention / queue drain ==")
        report["contention"] = run_contention(logger, args.base_port)
        print(render_text(report["contention"]["view"]))

        print("\n== Scenario 2: real monotonic expiry ==")
        report["expiry"] = run_expiry(logger, args.base_port + 10)
        print("verdicts:", report["expiry"]["verdicts"])
        print(render_text(report["expiry"]["view"]))

        print("\n== Scenario 3: forced termination + ledger replay ==")
        report["replay"] = run_replay(logger, args.base_port + 20)
        rec = report["replay"]["recovery"]
        print("unknown events:", [u["seq"] for u in rec["unknown_events"]])
        print("replay verify:", report["replay"]["verify"])
        print(render_text(report["replay"]["view"]))

        out = ROOT / "sample_logs" / "demo_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                  default=str), encoding="utf-8")
        print(f"\nstructured log: {log_path}")
        print(f"full report:    {out}")
        if args.ui:
            _launch_ui(report["replay"]["view"])
    finally:
        logger.close()
    return 0


def _launch_ui(view: dict[str, Any]) -> None:
    if not TK_AVAILABLE:
        print("\n[Tk] unavailable in this environment "
              f"({TK_IMPORT_ERROR}); text rendering used above. "
              "On Windows install official Python with tcl/tk enabled.")
        return
    dashboard = build_dashboard(lambda: view)
    if dashboard is None:
        print("\n[Tk] no display; text rendering used.")
        return
    dashboard.run()


def cmd_inject(args: argparse.Namespace) -> int:
    logger = StructuredLogger(ROOT / "sample_logs" / "injection.jsonl")
    try:
        if args.inject_timing:
            return inject_timing(logger)
        if args.inject_ref:
            return inject_ref(logger)
        if args.inject_snapshot:
            return inject_snapshot(logger, ROOT / "workdir" / "replay")
    finally:
        logger.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DistLock lease middleware runner")
    sub = p.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the three deterministic scenarios")
    demo.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    demo.add_argument("--ui", action="store_true",
                      help="open the Tkinter dashboard after the demo")
    demo.set_defaults(func=cmd_demo)

    inj = sub.add_parser("inject", help="inject a validation exception")
    grp = inj.add_mutually_exclusive_group(required=True)
    grp.add_argument("--inject-timing", action="store_true")
    grp.add_argument("--inject-ref", action="store_true")
    grp.add_argument("--inject-snapshot", action="store_true")
    inj.set_defaults(func=cmd_inject)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DistLockError as exc:
        logger = StructuredLogger(ROOT / "sample_logs" / "error.jsonl")
        logger.error(exc.code, "fatal", context=exc.context, message=exc.message)
        logger.close()
        print(f"FATAL {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
