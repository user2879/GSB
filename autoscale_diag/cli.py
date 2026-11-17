"""Command line entry points for the workbench."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .aggregation import choose_path
from .engine import Simulation, simulate
from .errors import BusinessError, E_LEDGER, E_REPLAY, E_USAGE, ledger_error
from .ledger import Ledger
from .replay import replay_ledger, verify_spec_against_ledger
from .reporting import build_report, write_report
from .spec import ClusterSpec, load_spec


def _print_json(payload: Any, *, stream: Any = None) -> None:
    target = stream or sys.stdout
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        file=target,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": report["run"]["run_id"],
        "declaration_version": report["run"]["declaration_version"],
        "status": report["run"]["status"],
        "path": report["run"]["path"],
        "window_capacity": report["run"]["window_capacity"],
        "peak_memory_estimate_bytes": report["run"]["peak_memory_estimate_bytes"],
        "committed_minutes": report["run"]["committed_minutes"],
        "minute_count": report["run"]["minute_count"],
        "budget": report["run"]["budget"],
        "summary": report["summary"],
        "ledger_digest": report["ledger_digest"],
        "business_codes": report["business_codes"],
        "exit_report_available": report["exit_report_available"],
    }


def _write_outputs(
    ledger: Ledger, run_id: str, spec: ClusterSpec, out_dir: str | None
) -> tuple[dict[str, Any], dict[str, str]]:
    report = build_report(ledger, run_id)
    files: dict[str, str] = {}
    if out_dir:
        directory = Path(out_dir)
        json_path = directory / "report.json"
        html_path = directory / "report.html"
        snapshot_path = directory / "snapshot.json"
        write_report(
            ledger,
            run_id,
            json_path=json_path,
            html_path=html_path,
            snapshot_path=snapshot_path,
        )
        files = {
            "report_json": str(json_path),
            "report_html": str(html_path),
            "snapshot_json": str(snapshot_path),
        }
    return report, files


def _process_minutes(
    spec: ClusterSpec,
    ledger: Ledger,
    run_id: str,
    path: str,
    *,
    states: dict[str, Any] | None = None,
    start_index: int = 0,
    stop_after: int | None = None,
) -> None:
    simulation = Simulation(spec, requested_path=path)
    if states is not None:
        simulation.states = states
    if start_index:
        simulation.prime(start_index)
    processed = 0
    for _ in range(start_index, spec.minute_count):
        minute_result = simulation.step()
        ledger.commit_minute(run_id, minute_result, spec.minute_count)
        processed += 1
        if stop_after is not None and processed >= stop_after:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(137)


def command_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    payload = {
        "ok": True,
        "business_code": "OK",
        "declaration_version": spec.declaration_version,
        "spec_sha256": spec.sha256(),
        "workloads": [item.id for item in spec.workloads],
        "minute_count": spec.minute_count,
        "global_replica_budget": spec.global_replica_budget,
        "sum_min_replicas": sum(item.min_replicas for item in spec.workloads),
        "auto_path": choose_path(
            spec.minute_count, spec.aggregation_threshold, "auto"
        ),
        "aggregation_threshold": spec.aggregation_threshold,
        "window_capacity": spec.window_size,
    }
    _print_json(payload)
    return 0


def command_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    path = choose_path(spec.minute_count, spec.aggregation_threshold, args.path)
    with Ledger(args.db) as ledger:
        run = ledger.create_run(
            spec,
            run_id=args.run_id,
            path=path,
            declaration_file=str(Path(args.spec)),
            replace=args.replace,
        )
        _process_minutes(
            spec,
            ledger,
            args.run_id,
            path,
            stop_after=args.stop_after,
        )
        ledger.finalize_run(args.run_id, spec.minute_count)
        report, files = _write_outputs(ledger, args.run_id, spec, args.out_dir)
        payload = {
            "ok": True,
            "business_codes": report["business_codes"],
            "run": _report_summary(report),
            "report_files": files,
        }
        if "replacement_backup" in run:
            payload["replacement_backup"] = run["replacement_backup"]
        _print_json(payload)
        return 0


def command_resume(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    with Ledger(args.db) as ledger:
        run = ledger.get_run(args.run_id)
        if run is None:
            raise ledger_error("run does not exist", "run_id", run_id=args.run_id)
        verify_spec_against_ledger(spec, run)
        committed_before = ledger.committed_count(args.run_id)
        if run["status"] != "complete":
            ledger.prepare_resume(args.run_id, spec.minute_count)
            prefix_replay = replay_ledger(
                ledger, spec, args.run_id, limit=committed_before
            )
            ledger.mark_running(args.run_id)
            states = ledger.latest_states(
                args.run_id, (workload.id for workload in spec.workloads)
            )
            _process_minutes(
                spec,
                ledger,
                args.run_id,
                run["path"],
                states=states,
                start_index=committed_before,
                stop_after=args.stop_after,
            )
            ledger.finalize_run(args.run_id, spec.minute_count)
        else:
            prefix_replay = replay_ledger(ledger, spec, args.run_id)
        report, files = _write_outputs(ledger, args.run_id, spec, args.out_dir)
        payload = {
            "ok": True,
            "business_codes": report["business_codes"],
            "run": _report_summary(report),
            "recovered_from_committed_minutes": committed_before,
            "prefix_replay": prefix_replay.to_dict(),
            "report_files": files,
        }
        _print_json(payload)
        return 0


def command_replay(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    with Ledger(args.db) as ledger:
        run = ledger.get_run(args.run_id)
        if run is None:
            raise ledger_error("run does not exist", "run_id", run_id=args.run_id)
        result = replay_ledger(ledger, spec, args.run_id)
        complete = run["status"] == "complete" and result.verified_minutes == spec.minute_count
        payload = result.to_dict()
        payload["exit_report_available"] = complete
        payload["business_codes"] = ["OK"] if complete else ["OK", "E_LEDGER"]
        payload["completion_note"] = (
            "all committed minutes verified; exit report may be produced"
            if complete
            else "verified committed prefix only; uncommitted minutes are unknown and must be rerun"
        )
        _print_json(payload)
        return 0


def command_report(args: argparse.Namespace) -> int:
    with Ledger(args.db) as ledger:
        run = ledger.get_run(args.run_id)
        if run is None:
            raise ledger_error("run does not exist", "run_id", run_id=args.run_id)
        report = build_report(ledger, args.run_id)
        if not report["exit_report_available"] and not args.allow_incomplete:
            raise BusinessError(
                E_LEDGER,
                "exit report is unavailable until the run is complete",
                "run.status",
                {
                    "status": run["status"],
                    "committed_minutes": report["run"]["committed_minutes"],
                    "minute_count": run["minute_count"],
                    "action": "resume and rerun from the first unknown minute",
                },
            )
        files: dict[str, str] = {}
        if args.json_out or args.html_out or args.snapshot_out:
            write_report(
                ledger,
                args.run_id,
                json_path=args.json_out,
                html_path=args.html_out,
                snapshot_path=args.snapshot_out,
            )
            if args.json_out:
                files["report_json"] = args.json_out
            if args.html_out:
                files["report_html"] = args.html_out
            if args.snapshot_out:
                files["snapshot_json"] = args.snapshot_out
        payload = _report_summary(report)
        payload["ok"] = report["exit_report_available"]
        payload["report_files"] = files
        _print_json(payload)
        return 0


def command_compare_paths(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    full = simulate(spec, forced_path="full")
    incremental = simulate(spec, forced_path="incremental")
    equal = full.decision_digest() == incremental.decision_digest()
    payload = {
        "ok": equal,
        "business_code": "OK" if equal else "E_REPLAY",
        "declaration_version": spec.declaration_version,
        "minute_count": spec.minute_count,
        "aggregation_threshold": spec.aggregation_threshold,
        "auto_selected": choose_path(spec.minute_count, spec.aggregation_threshold, "auto"),
        "window_capacity": spec.window_size,
        "full": {
            "decision_digest": full.decision_digest(),
            "peak_memory_estimate_bytes": full.peak_memory_estimate_bytes,
        },
        "incremental": {
            "decision_digest": incremental.decision_digest(),
            "peak_memory_estimate_bytes": incremental.peak_memory_estimate_bytes,
        },
        "byte_identical_decisions_and_metrics": equal,
    }
    _print_json(payload)
    return 0 if equal else 5


def command_doctor(args: argparse.Namespace) -> int:
    checks: dict[str, Any] = {}
    checks["python"] = {
        "version": sys.version.split()[0],
        "ok": sys.version_info >= (3, 11),
    }
    for module_name in ("json", "sqlite3", "numpy", "tkinter"):
        try:
            module = importlib.import_module(module_name)
            checks[module_name] = {
                "ok": True,
                "version": str(getattr(module, "__version__", "stdlib/Tk")),
            }
        except Exception as exc:
            checks[module_name] = {"ok": False, "error": str(exc)}
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".autoscale-write-check-", dir=Path.cwd(), delete=True
        ) as handle:
            handle.write(b"ok")
            handle.flush()
        writable = {"ok": True, "directory": str(Path.cwd())}
    except OSError as exc:
        writable = {"ok": False, "error": str(exc), "directory": str(Path.cwd())}
    checks["writable_workspace"] = writable
    core_ready = all(checks[item]["ok"] for item in ("python", "json", "sqlite3", "writable_workspace"))
    full_ready = core_ready and checks["numpy"]["ok"] and checks["tkinter"]["ok"]
    payload = {
        "ok": full_ready,
        "core_ready": core_ready,
        "full_ready": full_ready,
        "checks": checks,
        "note": (
            "core and CLI run without numpy; the requested full local Windows setup also needs numpy and Tk"
        ),
    }
    _print_json(payload)
    return 0 if full_ready or not args.strict else 1


def command_ui(args: argparse.Namespace) -> int:
    from . import ui

    argv = ["--db", args.db, "--run-id", args.run_id]
    if args.check:
        argv.append("--check")
    if args.require_tk:
        argv.append("--require-tk")
    return ui.main(argv)


def command_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    payload = run_demo(Path(args.out))
    _print_json(payload)
    return 0 if payload.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autoscale_diag",
        description="Local Windows autoscaling metrics diagnostic workbench",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a local declaration")
    validate.add_argument("spec")
    validate.set_defaults(func=command_validate)

    run = sub.add_parser("run", help="create and run a new ledger")
    run.add_argument("spec")
    run.add_argument("--db", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--path", choices=("auto", "full", "incremental"), default="auto")
    run.add_argument("--out-dir")
    run.add_argument("--replace", action="store_true")
    run.add_argument(
        "--stop-after",
        type=_positive_int,
        metavar="N",
        help="commit N minutes then simulate a forced process termination",
    )
    run.set_defaults(func=command_run)

    resume = sub.add_parser("resume", help="replay the committed prefix then continue")
    resume.add_argument("spec")
    resume.add_argument("--db", required=True)
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--out-dir")
    resume.add_argument("--stop-after", type=_positive_int, metavar="N")
    resume.set_defaults(func=command_resume)

    replay = sub.add_parser("replay", help="verify committed decisions by replay")
    replay.add_argument("spec")
    replay.add_argument("--db", required=True)
    replay.add_argument("--run-id", required=True)
    replay.set_defaults(func=command_replay)

    report = sub.add_parser("report", help="write stable JSON, HTML and snapshot files")
    report.add_argument("--db", required=True)
    report.add_argument("--run-id", required=True)
    report.add_argument("--json-out")
    report.add_argument("--html-out")
    report.add_argument("--snapshot-out")
    report.add_argument("--allow-incomplete", action="store_true")
    report.set_defaults(func=command_report)

    compare = sub.add_parser("compare-paths", help="prove full/incremental parity on one input")
    compare.add_argument("spec")
    compare.set_defaults(func=command_compare_paths)

    doctor = sub.add_parser("doctor", help="check Python, numpy, Tkinter, SQLite and writability")
    doctor.add_argument("--strict", action="store_true")
    doctor.set_defaults(func=command_doctor)

    ui = sub.add_parser("ui", help="launch the Tkinter diagnostic workbench")
    ui.add_argument("--db", default=str(Path("artifacts/demo/workbench.db")))
    ui.add_argument("--run-id", default="demo")
    ui.add_argument("--check", action="store_true")
    ui.add_argument("--require-tk", action="store_true")
    ui.set_defaults(func=command_ui)

    demo = sub.add_parser("demo", help="run the deterministic three-scenario demo")
    demo.add_argument("--out", default=str(Path("artifacts/demo")))
    demo.set_defaults(func=command_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BusinessError as exc:
        _print_json(exc.to_dict(), stream=sys.stderr)
        return {
            "E_SPEC": 2,
            "E_REPLAY": 3,
            "E_LEDGER": 4,
            "E_USAGE": 2,
        }.get(exc.code, 1)
    except KeyboardInterrupt:
        _print_json({"ok": False, "business_code": "E_LEDGER", "message": "interrupted"}, stream=sys.stderr)
        return 130
    except Exception as exc:
        _print_json(
            {
                "ok": False,
                "business_code": "E_USAGE",
                "message": "unhandled local error",
                "details": {"error": f"{type(exc).__name__}: {exc}"},
            },
            stream=sys.stderr,
        )
        return 1
