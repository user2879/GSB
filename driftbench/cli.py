from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from .engine import DiagnosticEngine, comparable_report
from .errors import ConfigAbort
from .fingerprint import canonical_json, digest
from .injections import load_injections, run_injection_suite
from .loading import load_workspace
from .samplegen import generate_sample
from .simulation import run_edit_sequence


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _aborted_report(exc: ConfigAbort) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "report_type": "diagnostic",
        "mode": "aborted",
        "aborted": True,
        "failures": [failure.to_dict() for failure in exc.failures],
    }


def _strip_simulation(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": result["schema_version"],
        "report_type": result["report_type"],
        "all_equal": result["all_equal"],
        "comparison_scope": result["comparison_scope"],
        "batches": result["batches"],
        "final_state_fingerprint": result["final_state_fingerprint"],
    }


def _concurrency_verification(state: Any) -> dict[str, Any]:
    serial_state = state.clone()
    serial_state.config = replace(serial_state.config, concurrency=1)
    parallel_state = state.clone()
    serial = DiagnosticEngine(full_rebuild=True, check_snapshot=False).run(serial_state)
    parallel = DiagnosticEngine(full_rebuild=True, check_snapshot=False).run(parallel_state)
    serial_core = comparable_report(serial)
    parallel_core = comparable_report(parallel)
    return {
        "serial_concurrency": 1,
        "parallel_concurrency": parallel_state.config.concurrency,
        "equal": canonical_json(serial_core) == canonical_json(parallel_core),
        "serial_digest": digest(serial_core),
        "parallel_digest": digest(parallel_core),
        "serial_max_observed_active": serial["concurrency"]["max_observed_active"],
        "parallel_max_observed_active": parallel["concurrency"]["max_observed_active"],
        "within_configured_parallelism": (
            parallel["concurrency"]["max_observed_active"]
            <= parallel_state.config.concurrency
        ),
        "timing_ms": {
            "serial_total": serial["timing_ms"]["total_ms"],
            "parallel_total": parallel["timing_ms"]["total_ms"],
        },
    }


def cmd_generate(args: argparse.Namespace) -> int:
    metadata = generate_sample(args.out)
    print(json.dumps({"generated": args.out, **metadata}, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        state = load_workspace(args.sample_dir)
    except ConfigAbort as exc:
        report = _aborted_report(exc)
        if args.output:
            _write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 2
    engine = DiagnosticEngine(
        full_rebuild=args.full,
        snapshot_dir=args.snapshot_dir,
        check_snapshot=bool(args.snapshot_dir),
    )
    report = engine.run(state)
    if args.output:
        _write_json(args.output, report)
    summary = {
        "output": args.output,
        "mode": report["mode"],
        "active_pair": report["active_pair"],
        "failures": len(report["failures"]),
        "reuse": report["reuse"]["counts"],
        "snapshot": report["snapshot"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if not any(f["severity"] == "ERROR" for f in report["failures"]) else 1


def cmd_simulate(args: argparse.Namespace) -> int:
    try:
        state = load_workspace(args.sample_dir)
    except ConfigAbort as exc:
        report = _aborted_report(exc)
        if args.verification:
            _write_json(args.verification, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 2
    snapshot_dir = args.snapshot_dir
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if not snapshot_dir:
        temporary = tempfile.TemporaryDirectory(prefix="driftbench-snapshots-")
        snapshot_dir = temporary.name
    try:
        result = run_edit_sequence(state, snapshot_dir=snapshot_dir)
        final_report = result["final_report"]
        verification = _strip_simulation(result)
        if args.report:
            _write_json(args.report, final_report)
        if args.verification:
            _write_json(args.verification, verification)
        injection_result = None
        injection_path = Path(args.sample_dir) / "injections.json"
        if injection_path.exists():
            injection_result = run_injection_suite(state, load_injections(injection_path))
            if args.injections:
                _write_json(args.injections, injection_result)
        concurrency_result = _concurrency_verification(result["final_state"])
        if args.concurrency:
            _write_json(args.concurrency, concurrency_result)
        summary = {
            "all_equal": result["all_equal"],
            "batches": len(verification["batches"]),
            "final_active_spec_version": final_report["active_spec_version"],
            "final_active_pair": final_report["active_pair"],
            "final_failures": len(final_report["failures"]),
            "injections_all_hit": injection_result["all_hit"] if injection_result else None,
            "concurrency_equal": concurrency_result["equal"],
            "concurrency_within_limit": concurrency_result["within_configured_parallelism"],
            "report": args.report,
            "verification": args.verification,
            "injections": args.injections,
            "concurrency": args.concurrency,
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if (
            result["all_equal"]
            and (injection_result is None or injection_result["all_hit"])
            and concurrency_result["equal"]
            and concurrency_result["within_configured_parallelism"]
        ) else 1
    finally:
        if temporary is not None:
            temporary.cleanup()


def cmd_injections(args: argparse.Namespace) -> int:
    try:
        state = load_workspace(args.sample_dir)
    except ConfigAbort as exc:
        report = _aborted_report(exc)
        if args.output:
            _write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 2
    result = run_injection_suite(state, load_injections(args.input))
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["all_hit"] else 1


def cmd_all(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot_dir = args.snapshot_dir or str(output / "snapshots")
    namespace = argparse.Namespace(
        sample_dir=args.sample_dir,
        snapshot_dir=snapshot_dir,
        report=str(output / "final_report.json"),
        verification=str(output / "verification.json"),
        injections=str(output / "injection_report.json"),
        concurrency=str(output / "concurrency_verification.json"),
    )
    return cmd_simulate(namespace)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="driftbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="write deterministic sample files")
    generate.add_argument("--out", default="samples")
    generate.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="run diagnostics on the initial sample state")
    run.add_argument("--sample-dir", default="samples")
    run.add_argument("--full", action="store_true", help="bypass all reuse")
    run.add_argument("--snapshot-dir")
    run.add_argument("--output")
    run.set_defaults(func=cmd_run)

    simulate = sub.add_parser("simulate", help="run edit sequence and compare each batch to full rebuild")
    simulate.add_argument("--sample-dir", default="samples")
    simulate.add_argument("--snapshot-dir")
    simulate.add_argument("--report", default="artifacts/final_report.json")
    simulate.add_argument("--verification", default="artifacts/verification.json")
    simulate.add_argument("--injections", default="artifacts/injection_report.json")
    simulate.add_argument("--concurrency", default="artifacts/concurrency_verification.json")
    simulate.set_defaults(func=cmd_simulate)

    injections = sub.add_parser("injections", help="run isolated fault injection scenarios")
    injections.add_argument("--sample-dir", default="samples")
    injections.add_argument("--input", default="samples/injections.json")
    injections.add_argument("--output", default="artifacts/injection_report.json")
    injections.set_defaults(func=cmd_injections)

    all_cmd = sub.add_parser("all", help="simulate, verify, inject, and emit all artifacts")
    all_cmd.add_argument("--sample-dir", default="samples")
    all_cmd.add_argument("--output-dir", default="artifacts")
    all_cmd.add_argument("--snapshot-dir")
    all_cmd.set_defaults(func=cmd_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
