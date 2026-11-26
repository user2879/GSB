"""Command line interface for the deterministic fixed-point laboratory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .cache import write_json_atomic
from .config import load_config
from .dataset import apply_commands, dataset_from_records, load_command_stream, load_samples_jsonl
from .errors import E_INPUT_INVALID, InputValidationError
from .hashing import canonical_sha256
from .pipeline import run_pipeline
from .samplegen import generate_samples, write_jsonl
from .snapshot import commit_snapshot


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _load_previous_ledger(path: Path) -> dict[str, dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {entry["artifact_id"]: entry for entry in data.get("entries", [])}
    except (json.JSONDecodeError, KeyError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fixed-lab")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-samples")
    generate.add_argument("--count", type=int, default=18)
    generate.add_argument("--seed", type=int, default=396)
    generate.add_argument("--out", default="data/samples.jsonl")

    validate = sub.add_parser("validate")
    validate.add_argument("--samples", required=True)
    validate.add_argument("--config", required=True)
    validate.add_argument("--edits")

    run = sub.add_parser("run")
    run.add_argument("--samples", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--edits")
    run.add_argument("--out-dir", default="runs/latest")
    run.add_argument("--cache-dir")
    run.add_argument("--ledger")
    run.add_argument("--snapshot-dir")
    run.add_argument("--workers", type=int)
    run.add_argument("--no-snapshot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-samples":
        path = write_jsonl(generate_samples(args.count, args.seed), args.out)
        _print_json({"status": "OK", "samples": args.count, "path": str(path)})
        return 0

    try:
        config = load_config(args.config)
    except InputValidationError as exc:
        _print_json({"status": E_INPUT_INVALID, "field": exc.field, "errors": exc.errors})
        return 2

    records, sample_errors = load_samples_jsonl(args.samples, config["columns"])
    command_list: list[dict[str, Any]] = []
    command_parse_errors: list[dict[str, Any]] = []
    if getattr(args, "edits", None):
        command_list, command_parse_errors = load_command_stream(args.edits)

    if args.command == "validate":
        command_results: list[dict[str, Any]] = []
        applied_command_errors = command_parse_errors
        if records and not sample_errors:
            dataset = dataset_from_records(config["columns"], records)
            _, _, command_results, apply_errors = apply_commands(dataset, config, command_list)
            applied_command_errors = command_parse_errors + apply_errors
        response = {
            "status": E_INPUT_INVALID if sample_errors or applied_command_errors else "OK",
            "valid_sample_count": len(records),
            "errors": sample_errors + applied_command_errors,
            "command_results": command_results,
        }
        _print_json(response)
        return 2 if response["status"] == E_INPUT_INVALID else 0

    dataset = dataset_from_records(config["columns"], records) if records else None
    command_results: list[dict[str, Any]] = []
    apply_errors: list[dict[str, Any]] = []
    if dataset is not None and args.edits:
        dataset, config, command_results, apply_errors = apply_commands(dataset, config, command_list)

    all_input_errors = sample_errors + command_parse_errors + apply_errors
    if dataset is None:
        _print_json({"status": E_INPUT_INVALID, "errors": all_input_errors})
        return 2

    ledger_path = Path(args.ledger or config["ledger_path"])
    previous = _load_previous_ledger(ledger_path)
    cache_dir = args.cache_dir if args.cache_dir is not None else config["cache_dir"]
    result = run_pipeline(
        dataset,
        config,
        cache_dir=cache_dir,
        previous_lineage=previous,
        worker_count=args.workers,
        input_errors=sample_errors,
        command_results=command_results,
        command_errors=command_parse_errors + apply_errors,
    )
    report = result["report"]
    trace = result["trace"]
    ledger_output = result["ledger"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "report.json", report)
    write_json_atomic(out_dir / "trace.json", trace)
    write_json_atomic(out_dir / "ledger.json", ledger_output)
    write_json_atomic(ledger_path, ledger_output)

    if not args.no_snapshot:
        snapshot_dir = Path(args.snapshot_dir or config["snapshot_dir"])
        version_hash = canonical_sha256(report)
        receipt = commit_snapshot(
            snapshot_dir,
            {"report.json": report, "trace.json": trace, "ledger.json": ledger_output},
            version_hash=version_hash,
        )
        report["snapshot"] = receipt
        write_json_atomic(out_dir / "report.json", report)

    _print_json({
        "status": report["status"],
        "out_dir": str(out_dir),
        "report": str(out_dir / "report.json"),
        "ledger": str(ledger_path),
        "snapshot": report.get("snapshot"),
        "ledger_summary": ledger_output["summary"],
        "input_error_count": len(all_input_errors),
    })
    return 2 if all_input_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
