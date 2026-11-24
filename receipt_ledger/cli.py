from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .generator import init_workspace
from .ports import check_worker_ports
from .runner import run_pipeline
from .runtime import SignalInterrupt, SignalGate
from .snapshot import diff_objects
from .config import validate_request_stream, validate_rules, validate_workers
from .utils import read_json, write_json
from .workspace import load_workspace


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    paths = init_workspace(root, seed=args.seed)
    result = run_pipeline(
        root / "data",
        root / "runs" / "reference",
        reset=True,
        check_ports=True,
        restart_each_command=False,
        jobs=args.jobs,
    )
    occupied = [check for check in result["report"]["port_checks"] if not check["free"]]
    summary = {
        "seed": args.seed,
        "data_dir": str(paths["data_dir"]),
        "reference_ledger": str(result["ledger_path"]),
        "reference_report": str(result["report_path"]),
        "port_checks": result["report"]["port_checks"],
        "receipt_counts": result["ledger"]["receipt_counts"],
        "business_code_counts": result["ledger"]["business_code_counts"],
    }
    _print_json(summary)
    if occupied:
        print("loopback port occupancy detected", file=sys.stderr)
        return 2
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    injection_config = None
    if args.injections:
        injection_config = Path(args.injections) if isinstance(args.injections, str) else Path(args.data) / "injections.json"
    gate = SignalGate()
    gate.install()
    try:
        result = run_pipeline(
            args.data,
            args.run,
            injection_config=injection_config,
            injection_ids=args.injection_id or None,
            jobs=args.jobs,
            restart_each_command=args.restart_each_command,
            reset=args.reset,
            check_ports=args.check_ports,
            max_commands=args.max_commands,
            signal_gate=gate,
        )
    except SignalInterrupt as exc:
        _print_json({"interrupted": True, "signal": exc.signum, "last_committed_snapshot_preserved": True})
        return 128 + exc.signum
    finally:
        gate.restore()
    _print_json({
        "ledger": str(result["ledger_path"]),
        "report": str(result["report_path"]),
        "command_cursor": result["ledger"]["command_cursor"],
        "summary": result["ledger"]["summary"],
        "business_code_counts": result["ledger"]["business_code_counts"],
        "recovery": result["recovery"],
    })
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    left = read_json(Path(args.left))
    right = read_json(Path(args.right))
    differences = diff_objects(left, right)
    if differences:
        _print_json({"equal": False, "difference_count": len(differences), "differences": differences[: args.limit]})
        return 1
    _print_json({"equal": True, "left": args.left, "right": args.right})
    return 0


def cmd_ports(args: argparse.Namespace) -> int:
    inputs, _texts, _hash, _injections = load_workspace(args.data)
    checks, failures = check_worker_ports(inputs["workers"])
    _print_json({"checks": checks, "failures": failures})
    return 0 if not failures else 2


def cmd_validate(args: argparse.Namespace) -> int:
    data = Path(args.data)
    inputs, _texts, _hash, _injections = load_workspace(data)
    report: dict[str, Any] = {
        "main_failures": inputs["failures"],
        "invalid_case_results": [],
    }
    invalid_dir = data / "invalid_cases"
    if invalid_dir.exists():
        workers_payload = read_json(invalid_dir / "workers.json")
        workers, worker_failures = validate_workers(workers_payload)
        worker_ids = {worker["worker_id"] for worker in workers} | {"W-OK"}
        stream_payload = read_json(invalid_dir / "request_stream.json")
        _requests, _batches, stream_failures = validate_request_stream(stream_payload, worker_ids)
        rules_payload = read_json(invalid_dir / "rules.json")
        _rules, rule_failures = validate_rules(rules_payload, worker_ids)
        for name, failures in (
            ("workers.json", worker_failures),
            ("request_stream.json", stream_failures),
            ("rules.json", rule_failures),
        ):
            report["invalid_case_results"].append({"file": name, "failures": failures})
    counts: dict[str, int] = {}
    for failure in report["main_failures"]:
        counts[failure["code"]] = counts.get(failure["code"], 0) + 1
    for case in report["invalid_case_results"]:
        for failure in case["failures"]:
            counts[failure["code"]] = counts.get(failure["code"], 0) + 1
    report["business_code_counts"] = dict(sorted(counts.items()))
    output = Path(args.output)
    write_json(output, report)
    _print_json({"output": str(output), "business_code_counts": report["business_code_counts"]})
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    init_workspace(root, seed=args.seed)
    data = root / "data"
    runs = root / "runs"

    reference = run_pipeline(data, runs / "reference", reset=True, jobs=1)
    incremental = run_pipeline(data, runs / "incremental-restart", reset=True, jobs=args.jobs,
                               restart_each_command=True)
    differences = diff_objects(reference["ledger"], incremental["ledger"])

    pending = run_pipeline(data, runs / "budget-pending", reset=True, max_commands=1, jobs=1)
    pending_summary = pending["ledger"]["summary"]
    pending_ok = pending_summary["pending_count"] > 0 and "E_NORMALIZE_BUDGET_PENDING" in pending["ledger"]["business_code_counts"]

    completed = run_pipeline(data, runs / "budget-pending", jobs=args.jobs)
    pending_then_full_differences = diff_objects(reference["ledger"], completed["ledger"])

    injected = run_pipeline(
        data,
        runs / "injections",
        reset=True,
        injection_config=data / "injections.json",
        jobs=1,
    )
    injection_results = injected["ledger"]["injection_results"]
    injections_ok = len(injection_results) == 4 and all(item.get("observed") for item in injection_results)

    threaded = run_pipeline(data, runs / "threaded", reset=True, jobs=4)
    threaded_differences = diff_objects(reference["ledger"], threaded["ledger"])

    verification = {
        "incremental_restart_matches_full": not differences,
        "incremental_differences": differences[:20],
        "budget_pending_demonstrated": pending_ok,
        "initial_pending_count": pending_summary["pending_count"],
        "pending_then_complete_matches_full": not pending_then_full_differences,
        "pending_differences": pending_then_full_differences[:20],
        "injections_observed": injections_ok,
        "injection_results": injection_results,
        "threaded_matches_single_thread": not threaded_differences,
        "threaded_differences": threaded_differences[:20],
        "reference_summary": reference["ledger"]["summary"],
        "reference_business_code_counts": reference["ledger"]["business_code_counts"],
    }
    write_json(runs / "verification.json", verification)
    diagnostic_report = dict(injected["report"])
    diagnostic_report["acceptance"] = {
        key: verification[key]
        for key in (
            "incremental_restart_matches_full",
            "budget_pending_demonstrated",
            "pending_then_complete_matches_full",
            "injections_observed",
            "threaded_matches_single_thread",
        )
    }
    diagnostic_report["comparison_artifacts"] = {
        "reference_ledger": str(reference["ledger_path"]),
        "incremental_ledger": str(incremental["ledger_path"]),
        "budget_completed_ledger": str(completed["ledger_path"]),
        "injection_report": str(injected["report_path"]),
    }
    write_json(runs / "diagnostic-report.json", diagnostic_report)
    _print_json(verification)
    ok = (
        not differences
        and pending_ok
        and not pending_then_full_differences
        and injections_ok
        and not threaded_differences
    )
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baseline", description="Deterministic receipt ledger baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="generate samples and reference ledger")
    init_parser.add_argument("--root", default=".")
    init_parser.add_argument("--seed", type=int, default=234)
    init_parser.add_argument("--jobs", type=int, default=4)
    init_parser.set_defaults(func=cmd_init)

    run_parser = sub.add_parser("run", help="run incremental ledger")
    run_parser.add_argument("--data", default="data")
    run_parser.add_argument("--run", default="runs/default")
    run_parser.add_argument("--injections", nargs="?", const=True, default=None)
    run_parser.add_argument("--injection-id", action="append", default=[])
    run_parser.add_argument("--jobs", type=int, default=4)
    run_parser.add_argument("--max-commands", type=int, default=None)
    run_parser.add_argument("--reset", action="store_true")
    run_parser.add_argument("--restart-each-command", action="store_true")
    run_parser.add_argument("--check-ports", action="store_true")
    run_parser.set_defaults(func=cmd_run)

    compare_parser = sub.add_parser("compare", help="field-by-field compare two ledger.json files")
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")
    compare_parser.add_argument("--limit", type=int, default=50)
    compare_parser.set_defaults(func=cmd_compare)

    ports_parser = sub.add_parser("ports", help="check loopback ports")
    ports_parser.add_argument("--data", default="data")
    ports_parser.set_defaults(func=cmd_ports)

    validate_parser = sub.add_parser("validate", help="validate main and deterministic invalid fixtures")
    validate_parser.add_argument("--data", default="data")
    validate_parser.add_argument("--output", default="runs/validation-report.json")
    validate_parser.set_defaults(func=cmd_validate)

    verify_parser = sub.add_parser("verify", help="run all local acceptance scenarios")
    verify_parser.add_argument("--root", default=".")
    verify_parser.add_argument("--seed", type=int, default=234)
    verify_parser.add_argument("--jobs", type=int, default=4)
    verify_parser.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
