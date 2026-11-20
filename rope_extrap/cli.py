from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .analysis import build_extended_report, diagnostic_consistency_view
from .engine import DiagnosticEngine, run_full_diagnosis
from .loader import load_position_config, load_samples
from .session import run_command_sequence
from .snapshot import SnapshotStore
from .state import state_from_sequence_result


def _write_json(path: str | Path | None, value: Any) -> None:
    if path is None:
        json.dump(value, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        sys.stdout.write("\n")
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_valid_inputs(config_path: str, samples_path: str):
    config, config_failures = load_position_config(config_path)
    if config is None:
        raise SystemExit(
            json.dumps(
                {"business_codes": sorted({failure.code for failure in config_failures}), "failures": [f.to_dict() for f in config_failures]},
                ensure_ascii=False,
            )
        )
    samples, sample_failures = load_samples(
        samples_path,
        dimension=config.dimension,
        target_length=config.extrapolation_target_length,
    )
    return config, samples, sample_failures


def command_generate(args: argparse.Namespace) -> int:
    from scripts.generate_samples import generate

    generate(Path(args.output))
    print(f"generated deterministic fixtures in {Path(args.output).resolve()}")
    return 0


def command_diagnose(args: argparse.Namespace) -> int:
    if args.commands:
        result = run_command_sequence(
            args.config,
            args.samples,
            args.commands,
            snapshot_dir=args.snapshot_dir,
            drain_pending=not args.stop_if_pending,
            drain_budget_ms=args.drain_budget_ms,
        )
        if result.get("final_report") is None:
            _write_json(args.out, result)
            return 2
        config, samples, state_errors = state_from_sequence_result(result)
        if config is None:
            _write_json(args.out, {**result, "state_errors": state_errors})
            return 2
        report = result["final_report"]
        if args.extended:
            report = build_extended_report(
                report,
                config,
                samples,
                include_sensitivity=not args.no_sensitivity,
            )
        if args.full_comparison:
            full_report = run_full_diagnosis(config, samples)
            report["incremental_full_consistency"] = {
                "equal_substantive_fields": (
                    diagnostic_consistency_view(result["final_report"])
                    == diagnostic_consistency_view(full_report)
                ),
                "incremental_summary": result["final_report"]["summary"],
                "full_summary": full_report["summary"],
            }
            if args.full_out:
                _write_json(args.full_out, full_report)
        _write_json(args.out, report)
        if args.responses_out:
            _write_json(args.responses_out, result["responses"])
        return 0

    config, samples, sample_failures = _load_valid_inputs(args.config, args.samples)
    report = run_full_diagnosis(config, samples, loader_failures=sample_failures)
    if args.extended:
        report = build_extended_report(
            report,
            config,
            samples,
            loader_failures=sample_failures,
            include_sensitivity=not args.no_sensitivity,
        )
    if args.snapshot_dir:
        engine = DiagnosticEngine(config, samples, loader_failures=sample_failures)
        plan = engine.run_full()
        full_report = engine.build_report(plan=plan)
        store = SnapshotStore(args.snapshot_dir)
        manifest = store.commit(
            engine,
            global_version=args.snapshot_version,
            completed_samples=[row["sample_id"] for row in full_report["samples"]],
        )
        report["snapshot_manifest"] = manifest
    _write_json(args.out, report)
    if args.responses_out:
        _write_json(args.responses_out, [])
    return 0


def command_faults(args: argparse.Namespace) -> int:
    config, config_failures = load_position_config(args.config)
    output: dict[str, Any] = {
        "config_failures": [failure.to_dict() for failure in config_failures],
    }
    if config is not None:
        samples, sample_failures = load_samples(
            args.samples,
            dimension=config.dimension,
            target_length=config.extrapolation_target_length,
        )
        output.update(
            {
                "valid_sample_ids": sorted(samples),
                "sample_failures": [failure.to_dict() for failure in sample_failures],
                "business_code_counts": _count_codes(config_failures + sample_failures),
            }
        )
    _write_json(args.out, output)
    return 0


def command_snapshot_load(args: argparse.Namespace) -> int:
    store = SnapshotStore(args.snapshot_dir)
    result = store.load(
        expected_version=args.version,
        expected_config_hash=args.config_hash,
    )
    _write_json(
        args.out,
        {
            "ok": result.ok,
            "manifest": result.manifest,
            "entry_count": len(result.entries),
            "failures": [failure.to_dict() for failure in result.failures],
        },
    )
    return 0 if result.ok else 3


def _count_codes(failures: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        counts[failure.code] = counts.get(failure.code, 0) + 1
    return dict(sorted(counts.items()))


def command_verify(args: argparse.Namespace) -> int:
    from scripts.verify_acceptance import main as verify_main

    verify_main(Path(args.output_dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rope-extrap",
        description="Deterministic RoPE extrapolation diagnostics with incremental cache and latency budgets",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic sample fixtures")
    generate.add_argument("--output", default="samples")
    generate.set_defaults(handler=command_generate)

    diagnose = subparsers.add_parser("diagnose", help="run one-shot or command-sequence diagnostics")
    diagnose.add_argument("--config", required=True)
    diagnose.add_argument("--samples", required=True)
    diagnose.add_argument("--commands")
    diagnose.add_argument("--snapshot-dir")
    diagnose.add_argument("--snapshot-version", type=int, default=1)
    diagnose.add_argument("--out")
    diagnose.add_argument("--responses-out")
    diagnose.add_argument("--full-out")
    diagnose.add_argument("--extended", action="store_true", help="add all-scheme and sensitivity statistics")
    diagnose.add_argument("--no-sensitivity", action="store_true")
    diagnose.add_argument("--full-comparison", action="store_true")
    diagnose.add_argument("--stop-if-pending", action="store_true")
    diagnose.add_argument("--drain-budget-ms", type=int, default=120_000)
    diagnose.set_defaults(handler=command_diagnose)

    faults = subparsers.add_parser("faults", help="exercise loader validation and fault isolation")
    faults.add_argument("--config", required=True)
    faults.add_argument("--samples", required=True)
    faults.add_argument("--out")
    faults.set_defaults(handler=command_faults)

    snapshot_load = subparsers.add_parser("snapshot-load", help="load and validate a snapshot version")
    snapshot_load.add_argument("--snapshot-dir", required=True)
    snapshot_load.add_argument("--version", type=int, required=True)
    snapshot_load.add_argument("--config-hash", required=True)
    snapshot_load.add_argument("--out")
    snapshot_load.set_defaults(handler=command_snapshot_load)

    verify = subparsers.add_parser("verify", help="run the complete acceptance verification")
    verify.add_argument("--output-dir", default="artifacts")
    verify.set_defaults(handler=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

