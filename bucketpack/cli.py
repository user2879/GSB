"""Command-line entry points for validation, runs, demos, recovery and faults."""

from __future__ import annotations

import argparse
import os
import socket
import sqlite3
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .canonical import dumps, write_json
from .changes import load_changes
from .engine import graph_view, run_full
from .errors import (
    E_GUI,
    E_INTERNAL,
    E_LEDGER,
    E_REPLAY,
    BusinessError,
    fail,
)
from .faults import INJECTION_CHOICES, apply_injections
from .graph import build_graph
from .incremental import run_incremental
from .ledger import (
    Ledger,
    ledger_summary,
    schedule_map_from_state,
)
from .logging_utils import StructuredLogger
from .numeric import NP_VERSION
from .spec import load_spec

DEFAULT_LOG = "artifacts/logs/bucketpack.jsonl"
DEFAULT_LEDGER = "artifacts/ledger.sqlite"
DEFAULT_RUN_RESULT = "artifacts/run_result.json"


def _print_json(value: Any) -> None:
    print(dumps(value))


def _next_batch_id(ledger: Ledger, spec_digest: str) -> str:
    existing = {item["batch_id"] for item in ledger.read_batches()}
    base = f"batch-{spec_digest[:12]}"
    if base not in existing:
        return base
    index = 2
    while f"{base}-{index}" in existing:
        index += 1
    return f"{base}-{index}"


def _result_summary(state: Any, out_path: str | None) -> dict[str, Any]:
    return {
        "ok": True,
        "spec_digest": state.result["spec_digest"],
        "content_digest": state.result["content_digest"],
        "pack_count": len(state.packs),
        "row_count": state.result["merged"]["global"]["row_count"],
        "width_used": state.result["merged"]["global"]["width_used"],
        "max_active_width": state.result["schedule"]["max_active_width"],
        "global_width_budget": state.result["schedule"]["global_width_budget"],
        "cost_total_micros": state.result["cost_ledger"]["total_cost_micros"],
        "cost_within_budget": state.result["cost_ledger"]["within_budget"],
        "result_file": out_path,
    }


def _run_result(args: argparse.Namespace, logger: StructuredLogger) -> int:
    spec = load_spec(args.spec)
    spec, overrides, ledger_failure = apply_injections(spec, args.inject)
    logger.emit(
        "RUN_START",
        "pipeline run started",
        spec=str(args.spec),
        spec_digest=spec.digest,
        injections=sorted(set(args.inject)),
    )
    ledger_report = None
    if args.ledger:
        with Ledger(args.ledger, fail_on_write=ledger_failure) as ledger:
            batch_id = args.batch_id or _next_batch_id(ledger, spec.digest)
            committed = {"count": 0}

            def on_plans_ready(plans: Any) -> None:
                ledger.begin_batch_plans(
                    batch_id,
                    plans,
                    spec_digest=spec.digest,
                )
                if args.crash_after is not None and args.crash_after <= 0:
                    os._exit(17)

            def on_pack_complete(pack: Any, entry: Any) -> None:
                ledger.commit_pack(batch_id, pack, entry)
                committed["count"] += 1
                logger.emit(
                    "PACK_COMMITTED",
                    "pack committed in its own transaction",
                    pack_id=pack.pack_id,
                    start_ms=entry.start_ms,
                    end_ms=entry.end_ms,
                )
                if args.crash_after is not None and committed["count"] >= args.crash_after:
                    os._exit(17)

            state = run_full(
                spec,
                value_overrides=overrides,
                on_plans_ready=on_plans_ready,
                on_pack_complete=on_pack_complete,
            )
            digest_value = ledger.finish_batch(
                batch_id, snapshot_digest=str(state.result["content_digest"])
            )
            ledger_report = {
                "batch_id": batch_id,
                "status": "complete",
                "content_digest": digest_value,
                "packs": len(state.packs),
            }
            logger.emit("BATCH_COMPLETE", "ledger batch completed", report=ledger_report)
    else:
        state = run_full(spec, value_overrides=overrides)
    if args.result_out:
        write_json(args.result_out, state.result)
    summary = _result_summary(state, args.result_out)
    summary["ledger"] = ledger_report
    _print_json(summary)
    return 0


def _incremental(args: argparse.Namespace, logger: StructuredLogger) -> int:
    spec = load_spec(args.spec)
    changes = load_changes(args.changes, spec)
    logger.emit("INCREMENTAL_START", "incremental run started", spec_digest=spec.digest)
    baseline = run_full(spec)
    outcome = run_incremental(spec, changes, baseline=baseline)
    ledger_report = None
    if args.ledger:
        with Ledger(args.ledger) as ledger:
            batch_id = args.batch_id or _next_batch_id(ledger, outcome.state.spec.digest)
            ledger_report = ledger.persist_state(outcome.state, batch_id)
            logger.emit("BATCH_COMPLETE", "incremental ledger batch completed", report=ledger_report)
    if args.result_out:
        write_json(args.result_out, outcome.state.result)
    if args.report_out:
        write_json(args.report_out, outcome.report)
    summary = _result_summary(outcome.state, args.result_out)
    summary.update(
        {
            "mode": "incremental",
            "report_file": args.report_out,
            "incremental_equals_full": outcome.report["equivalence"]["equal"],
            "invalidated_bucket_columns": outcome.report["invalidation"]["bucket_columns"],
            "invalidated_packs": outcome.report["invalidation"]["packs"],
            "recomputed_bucket_columns": outcome.report["recomputed"]["bucket_columns"],
            "recomputed_packs": outcome.report["recomputed"]["packs"],
            "reused_packs": outcome.report["recomputed"]["reused_packs"],
            "ledger": ledger_report,
        }
    )
    _print_json(summary)
    return 0


def _recover(args: argparse.Namespace, logger: StructuredLogger) -> int:
    spec = load_spec(args.spec)
    state = run_full(spec)
    with Ledger(args.ledger) as ledger:
        batches = ledger.read_batches()
        if not batches:
            fail(E_LEDGER, "$.ledger.batches", "ledger contains no batches", ledger_path=str(args.ledger))
        batch_id = args.batch_id or batches[-1]["batch_id"]
        schedule = schedule_map_from_state(state)
        logger.emit("RECOVERY_START", "ledger recovery started", batch_id=batch_id)
        report = ledger.recover_batch(
            batch_id,
            state.packs,
            spec_digest=spec.digest,
            snapshot_digest=state.result["content_digest"],
            schedule=schedule,
        )
        rerun: list[str] = []
        if report.get("requires_rerun"):
            by_id = {pack.pack_id: pack for pack in state.packs}
            for pack_id in report["unknown_packs"]:
                pack = by_id.get(pack_id)
                if pack is None:
                    fail(
                        E_REPLAY,
                        f"$.ledger.packs[{pack_id}]",
                        "unknown pack is absent from the replay snapshot",
                        first_differing_pack_id=pack_id,
                    )
                ledger.commit_pack(batch_id, pack, schedule.get(pack_id))
                rerun.append(pack_id)
                logger.emit("PACK_REPLAYED", "unknown pack replayed", pack_id=pack_id)
            ledger.finish_batch(batch_id, snapshot_digest=state.result["content_digest"])
            report = dict(report)
            report["status"] = "recovered"
            report["rerun_packs"] = rerun
            report["requires_rerun"] = False
        report = dict(report)
        report["ledger"] = ledger_summary(ledger)
        if args.report_out:
            write_json(args.report_out, report)
        logger.emit("RECOVERY_COMPLETE", "ledger recovery completed", batch_id=batch_id, status=report["status"])
        _print_json({"ok": True, **report})
    return 0


def _validate(args: argparse.Namespace, logger: StructuredLogger) -> int:
    spec = load_spec(args.spec)
    graph = build_graph(spec)
    payload = {
        "ok": True,
        "spec_digest": spec.digest,
        "sources": [
            {
                "id": source.id,
                "rows": source.rows,
                "primary_key": source.primary_key,
                "columns": [column.name for column in source.columns],
            }
            for source in spec.sources
        ],
        "graph": graph_view(spec, graph),
    }
    if args.out:
        write_json(args.out, payload)
    logger.emit("VALIDATE_OK", "specification and graph are valid", spec_digest=spec.digest)
    _print_json(payload)
    return 0


def _preflight(args: argparse.Namespace, logger: StructuredLogger) -> int:
    checks: dict[str, Any] = {}
    checks["python"] = {
        "ok": sys.version_info >= (3, 10),
        "version": sys.version.split()[0],
        "required": ">=3.10",
    }
    checks["sqlite3"] = {"ok": True, "version": sqlite3.sqlite_version}
    try:
        import tkinter

        checks["tkinter"] = {"ok": True, "version": str(tkinter.TkVersion)}
    except Exception as exc:
        checks["tkinter"] = {"ok": False, "error": str(exc), "headless_ok": True}
    checks["numpy"] = (
        {"ok": True, "version": NP_VERSION}
        if NP_VERSION is not None
        else {
            "ok": False,
            "error": "numpy is not installed; deterministic pure-Python fallback is active",
            "headless_ok": True,
        }
    )
    localhost_ok = True
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((args.host, args.port))
        bound = sock.getsockname()
        checks["localhost"] = {"ok": True, "host": args.host, "port": bound[1]}
    except OSError as exc:
        localhost_ok = False
        checks["localhost"] = {"ok": False, "host": args.host, "port": args.port, "error": str(exc)}
    finally:
        sock.close()
    sqlite_path = Path(args.sqlite_path)
    sqlite_ok = True
    try:
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(sqlite_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS __preflight__(v INTEGER)")
            conn.execute("INSERT INTO __preflight__(v) VALUES(1)")
            conn.execute("DELETE FROM __preflight__")
            conn.commit()
        checks["sqlite_path"] = {"ok": True, "path": str(sqlite_path.resolve())}
    except Exception as exc:
        sqlite_ok = False
        checks["sqlite_path"] = {"ok": False, "path": str(sqlite_path), "error": str(exc)}
    ok = (
        checks["python"]["ok"]
        and checks["sqlite3"]["ok"]
        and localhost_ok
        and sqlite_ok
    )
    if args.strict:
        ok = ok and checks["tkinter"]["ok"] and checks["numpy"]["ok"]
    payload = {"ok": ok, "checks": checks, "headless": not checks["tkinter"]["ok"]}
    if args.out:
        write_json(args.out, payload)
    logger.emit("PREFLIGHT", "preflight completed", ok=ok)
    _print_json(payload)
    return 0 if ok or not args.strict else 2


def _fault_probe(name: str, fn: Any) -> dict[str, Any]:
    try:
        fn()
    except BusinessError as exc:
        return {
            "injection": name,
            "ok": True,
            "code": exc.code,
            "path": exc.path,
            "message": exc.message,
            "details": dict(exc.details),
        }
    return {"injection": name, "ok": False, "code": None, "message": "expected failure did not occur"}


def _demo(args: argparse.Namespace, logger: StructuredLogger) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = load_spec(args.spec)
    logger.emit("DEMO_START", "demo started", spec_digest=spec.digest)

    baseline = run_full(spec)
    write_json(out_dir / "baseline_result.json", baseline.result)

    incremental_reports: list[dict[str, Any]] = []
    sample_dir = Path(args.spec).resolve().parent
    change_files = sorted(sample_dir.glob("changes_*.json"))
    for change_path in change_files:
        if change_path.name == "changes_ref_error.json":
            continue
        changes = load_changes(change_path, spec)
        outcome = run_incremental(spec, changes, baseline=baseline)
        suffix = change_path.stem.removeprefix("changes_")
        write_json(out_dir / f"incremental_{suffix}_result.json", outcome.state.result)
        write_json(out_dir / f"incremental_{suffix}_report.json", outcome.report)
        incremental_reports.append(
            {
                "changes_file": str(change_path),
                "equal": outcome.report["equivalence"]["equal"],
                "invalidated_buckets": outcome.report["invalidation"]["bucket_columns"],
                "invalidated_packs": outcome.report["invalidation"]["packs"],
                "recomputed_buckets": outcome.report["recomputed"]["bucket_columns"],
                "recomputed_packs": outcome.report["recomputed"]["packs"],
                "reused_packs": outcome.report["recomputed"]["reused_packs"],
            }
        )

    ref_error = {"ok": False}
    try:
        bad_changes = load_changes(sample_dir / "changes_ref_error.json", spec)
        run_incremental(spec, bad_changes, baseline=baseline)
    except BusinessError as exc:
        ref_error = {
            "ok": exc.code == "E_REF",
            "code": exc.code,
            "path": exc.path,
            "referrers": exc.details.get("referrers", []),
        }

    ledger_path = out_dir / "ledger.sqlite"
    if ledger_path.exists():
        ledger_path.unlink()
    with Ledger(ledger_path) as ledger:
        persisted = ledger.persist_state(baseline, "demo-baseline")
        recovered = ledger.recover_batch(
            "demo-baseline",
            baseline.packs,
            spec_digest=spec.digest,
            snapshot_digest=baseline.result["content_digest"],
            schedule=schedule_map_from_state(baseline),
        )
        ledger_batches = ledger.read_batches()

    crash_path = out_dir / "crash_ledger.sqlite"
    if crash_path.exists():
        crash_path.unlink()
    with Ledger(crash_path) as ledger:
        ledger.begin_batch(
            "demo-crash",
            baseline.packs,
            spec_digest=spec.digest,
            snapshot_digest=baseline.result["content_digest"],
        )
        schedule = schedule_map_from_state(baseline)
        for pack in baseline.packs[:2]:
            ledger.commit_pack("demo-crash", pack, schedule.get(pack.pack_id))
    with Ledger(crash_path) as ledger:
        partial = ledger.recover_batch(
            "demo-crash",
            baseline.packs,
            spec_digest=spec.digest,
            snapshot_digest=baseline.result["content_digest"],
            schedule=schedule,
        )
        for pack_id in partial["unknown_packs"]:
            pack = next(item for item in baseline.packs if item.pack_id == pack_id)
            ledger.commit_pack("demo-crash", pack, schedule.get(pack_id))
        ledger.finish_batch("demo-crash", snapshot_digest=baseline.result["content_digest"])
        recovered_crash = ledger.recover_batch(
            "demo-crash",
            baseline.packs,
            spec_digest=spec.digest,
            snapshot_digest=baseline.result["content_digest"],
            schedule=schedule,
        )

    replay_path = out_dir / "replay_ledger.sqlite"
    if replay_path.exists():
        replay_path.unlink()
    with Ledger(replay_path) as ledger:
        ledger.persist_state(baseline, "demo-replay")
    with sqlite3.connect(replay_path) as conn:
        conn.execute("UPDATE ledger_meta SET value='0' WHERE key='format_version'")
        conn.commit()
    replay_report: dict[str, Any]
    with Ledger(replay_path) as ledger:
        try:
            ledger.recover_batch(
                "demo-replay",
                baseline.packs,
                spec_digest=spec.digest,
                snapshot_digest=baseline.result["content_digest"],
                schedule=schedule_map_from_state(baseline),
            )
            replay_report = {"ok": False, "message": "expected E_REPLAY"}
        except BusinessError as exc:
            replay_report = {
                "ok": exc.code == E_REPLAY,
                "code": exc.code,
                "path": exc.path,
                "first_differing_pack_id": exc.details.get("first_differing_pack_id"),
                "replacement_format_version": ledger.format_version(),
                "replacement_batches": ledger.read_batches(),
            }

    injections: list[dict[str, Any]] = []

    def width_probe() -> None:
        injected_spec, overrides, _ = apply_injections(spec, ["width"])
        run_full(injected_spec, value_overrides=overrides)

    def bucket_probe() -> None:
        injected_spec, overrides, _ = apply_injections(spec, ["bucket"])
        run_full(injected_spec, value_overrides=overrides)

    def cycle_probe() -> None:
        injected_spec, overrides, _ = apply_injections(spec, ["cycle"])
        run_full(injected_spec, value_overrides=overrides)

    def ledger_probe() -> None:
        ledger_fault_path = out_dir / "ledger_fault.sqlite"
        if ledger_fault_path.exists():
            ledger_fault_path.unlink()
        with Ledger(ledger_fault_path, fail_on_write=True) as ledger:
            ledger.persist_state(baseline, "fault")

    for name, probe in (
        ("width", width_probe),
        ("bucket", bucket_probe),
        ("cycle", cycle_probe),
        ("ledger", ledger_probe),
    ):
        result = _fault_probe(name, probe)
        injections.append(result)
        logger.emit(
            result.get("code") or "INJECTION_MISSED",
            f"fault injection probe: {name}",
            level="INFO" if result["ok"] else "ERROR",
            injection=result.get("injection"),
            ok=result.get("ok"),
            path=result.get("path"),
            details=result.get("details", {}),
            observed_message=result.get("message"),
        )

    try:
        import tkinter  # noqa: F401

        gui_ready = True
    except Exception:
        gui_ready = False
    checks = {
        "incremental_equals_full": all(item["equal"] for item in incremental_reports),
        "ref_deletion_rejected": ref_error["ok"],
        "merge_equals_direct": baseline.result["merged"] == baseline.result["direct"],
        "schedule_within_global_budget": baseline.result["schedule"]["max_active_width"]
        <= baseline.result["schedule"]["global_width_budget"],
        "cost_within_budget": baseline.result["cost_ledger"]["within_budget"],
        "ledger_recovered": recovered["status"] == "complete",
        "crash_recovery_complete": recovered_crash["status"] == "complete",
        "replay_replacement": replay_report.get("ok", False),
        "fault_codes": {item["injection"]: item.get("code") for item in injections},
    }
    report = {
        "ok": all(
            [
                checks["incremental_equals_full"],
                checks["ref_deletion_rejected"],
                checks["merge_equals_direct"],
                checks["schedule_within_global_budget"],
                checks["cost_within_budget"],
                checks["ledger_recovered"],
                checks["crash_recovery_complete"],
                checks["replay_replacement"],
                all(item["ok"] for item in injections),
            ]
        ),
        "environment": {
            "python": sys.version.split()[0],
            "sqlite3": sqlite3.sqlite_version,
            "numpy": NP_VERSION,
            "tkinter_available": gui_ready,
        },
        "baseline": {
            "spec_digest": spec.digest,
            "content_digest": baseline.result["content_digest"],
            "pack_count": len(baseline.packs),
            "max_active_width": baseline.result["schedule"]["max_active_width"],
            "global_width_budget": baseline.result["schedule"]["global_width_budget"],
            "total_cost_micros": baseline.result["cost_ledger"]["total_cost_micros"],
            "within_budget": baseline.result["cost_ledger"]["within_budget"],
        },
        "incremental": incremental_reports,
        "ref_error": ref_error,
        "ledger": {
            "persist": persisted,
            "recover": recovered,
            "batches": ledger_batches,
        },
        "crash_recovery": {
            "partial_unknown_packs": partial["unknown_packs"],
            "verified_packs": partial["verified_packs"],
            "final": recovered_crash,
        },
        "replay": replay_report,
        "injections": injections,
        "checks": checks,
    }
    write_json(out_dir / "demo_report.json", report)
    logger.emit("DEMO_COMPLETE", "demo completed", ok=report["ok"], report=str(out_dir / "demo_report.json"))
    _print_json(
        {
            "ok": report["ok"],
            "report": str(out_dir / "demo_report.json"),
            "checks": checks,
        }
    )
    return 0 if report["ok"] else 2


def _gui(args: argparse.Namespace, logger: StructuredLogger) -> int:
    try:
        from .gui import launch_gui
    except Exception as exc:
        fail(E_GUI, "gui", "Tkinter GUI is not available", error=str(exc))
    launch_gui(
        spec_path=args.spec,
        changes_path=args.changes,
        ledger_path=args.ledger,
        log_path=args.log_file,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bucketpack",
        description="Deterministic bucket expansion, budgeted packing and SQLite ledger.",
    )
    parser.add_argument("--log-file", default=DEFAULT_LOG, help="structured JSON-lines log path")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(parser_item: argparse.ArgumentParser) -> None:
        parser_item.add_argument(
            "--log-file",
            default=argparse.SUPPRESS,
            help="structured JSON-lines log path (also accepted before the subcommand)",
        )

    pre = sub.add_parser("preflight", help="check Python, sqlite3, tkinter, numpy and local write access")
    add_common(pre)
    pre.add_argument("--sqlite-path", default=DEFAULT_LEDGER)
    pre.add_argument("--host", default="127.0.0.1")
    pre.add_argument("--port", type=int, default=0)
    pre.add_argument("--strict", action="store_true")
    pre.add_argument("--out")
    pre.set_defaults(handler=_preflight)

    val = sub.add_parser("validate", help="validate spec and dependency graph")
    add_common(val)
    val.add_argument("--spec", required=True)
    val.add_argument("--out")
    val.set_defaults(handler=_validate)

    run = sub.add_parser("run", help="run a full pipeline and persist the ledger")
    add_common(run)
    run.add_argument("--spec", required=True)
    run.add_argument("--ledger", default=DEFAULT_LEDGER)
    run.add_argument("--batch-id")
    run.add_argument("--result-out", default=DEFAULT_RUN_RESULT)
    run.add_argument("--inject", action="append", choices=INJECTION_CHOICES, default=[])
    run.add_argument("--crash-after", type=int)
    run.set_defaults(handler=_run_result)

    inc = sub.add_parser("incremental", help="apply a change set and verify against a full rebuild")
    add_common(inc)
    inc.add_argument("--spec", required=True)
    inc.add_argument("--changes", required=True)
    inc.add_argument("--ledger")
    inc.add_argument("--batch-id")
    inc.add_argument("--result-out", default="artifacts/incremental_result.json")
    inc.add_argument("--report-out", default="artifacts/incremental_report.json")
    inc.set_defaults(handler=_incremental)

    rec = sub.add_parser("recover", help="recover committed packs and replay unknown packs")
    add_common(rec)
    rec.add_argument("--spec", required=True)
    rec.add_argument("--ledger", default=DEFAULT_LEDGER)
    rec.add_argument("--batch-id")
    rec.add_argument("--report-out", default="artifacts/recovery_report.json")
    rec.set_defaults(handler=_recover)

    demo = sub.add_parser("demo", help="run the complete local acceptance demo")
    add_common(demo)
    demo.add_argument("--spec", default="samples/spec.json")
    demo.add_argument("--out-dir", default="artifacts/demo")
    demo.set_defaults(handler=_demo)

    gui = sub.add_parser("gui", help="launch the optional Tkinter inspection UI")
    add_common(gui)
    gui.add_argument("--spec", default="samples/spec.json")
    gui.add_argument("--changes")
    gui.add_argument("--ledger", default=DEFAULT_LEDGER)
    gui.set_defaults(handler=_gui)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = StructuredLogger(args.log_file)
    try:
        return int(args.handler(args, logger))
    except BusinessError as exc:
        logger.emit(
            exc.code,
            exc.message,
            level="ERROR",
            path=exc.path,
            details=dict(exc.details),
        )
        print(dumps(exc.to_dict()), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        logger.emit("E_INTERRUPTED", "interrupted", level="ERROR")
        print(dumps({"ok": False, "code": "E_INTERRUPTED", "message": "interrupted"}), file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        logger.emit(
            E_INTERNAL,
            "unhandled internal error",
            level="ERROR",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        print(
            dumps(
                {
                    "ok": False,
                    "code": E_INTERNAL,
                    "path": "$",
                    "message": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
