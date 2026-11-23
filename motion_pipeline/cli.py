"""Command-line entry point for deterministic pipeline runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .config import load_config
from .runner import PipelineRunner, RunRequest
from .samples import load_sequences
from .stable import canonical_json, public_failure, sort_failures


def _write_document(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.tmp-{os.getpid()}"
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(canonical_json(document))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _write_load_failure(
    output_dir: Path,
    failures: list[dict[str, Any]],
    *,
    mode: str,
    start_ns: int,
) -> None:
    failures = sort_failures(failures)
    end_ns = time.monotonic_ns()
    codes = sorted({entry["business_code"] for entry in failures})
    empty_predictions = {
        "schema_version": 1,
        "scale_tier": None,
        "selected_strategy": None,
        "sequences": [],
        "failures": failures,
    }
    empty_decision = {
        "schema_version": 1,
        "scale_tier": None,
        "metrics": {},
        "candidates": [],
        "selected": None,
        "reason": f"{mode} validation failed; no strategy selection was performed",
    }
    empty_ledger = {
        "schema_version": 1,
        "scale_tier": None,
        "execution_event_log": "cache_execution.json",
        "entries": [],
    }
    empty_execution = {
        "schema_version": 1,
        "scale_tier": None,
        "events": [],
        "hit_set": [],
        "recompute_set": [],
    }
    diagnostic = {
        "schema_version": 1,
        "selected_strategy": None,
        "quantization_level": None,
        "decision_reason": empty_decision["reason"],
        "scale_metrics": {},
        "stage_cache": {
            stage: {"hit": 0, "recompute": 0}
            for stage in ("encode", "motion", "predict", "decode")
        },
        "stage_cache_totals": {"hit": 0, "recompute": 0},
        "per_sequence_prediction_error": [],
        "monotonic_clock": {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "elapsed_ms": round((end_ns - start_ns) / 1_000_000.0, 6),
            "stage_elapsed_ms": {},
        },
        "thread_pool": {"configured_workers": None, "observed_max_concurrency": 0},
        "latency_budget_ms": None,
        "latency_budget_met": None,
        "failure_business_codes": codes,
        "failures": failures,
        "snapshot": {
            "global_version": None,
            "config_hash": None,
            "completed_sequences": [],
            "stage_versions": {},
            "stale_events": [],
            "stale_business_codes": [],
        },
    }
    for name, document in (
        ("predictions.json", empty_predictions),
        ("decision_record.json", empty_decision),
        ("cache_ledger.json", empty_ledger),
        ("cache_execution.json", empty_execution),
        ("diagnostic_report.json", diagnostic),
        ("failures.json", failures),
    ):
        _write_document(output_dir / name, document)


def cmd_run(args: argparse.Namespace) -> int:
    start_ns = time.monotonic_ns()
    output_dir = Path(args.output_dir)
    config_result = load_config(args.config)
    if not config_result.ok:
        _write_load_failure(
            output_dir, config_result.failures, mode="configuration", start_ns=start_ns
        )
        print(
            json.dumps(
                {
                    "status": "failed",
                    "business_codes": sorted(
                        {item["business_code"] for item in config_result.failures}
                    ),
                    "failures": config_result.failures,
                    "output_dir": str(output_dir),
                },
                sort_keys=True,
            )
        )
        return 2
    config = config_result.config
    assert config is not None
    if args.tier not in config.scale_tiers:
        failure = public_failure(
            "E_CONFIG_INVALID",
            "__config__",
            f"requested scale tier {args.tier} is not in {list(config.scale_tiers)}",
            field="scale_tiers",
        )
        _write_load_failure(output_dir, [failure], mode="scale-tier", start_ns=start_ns)
        print(
            json.dumps(
                {
                    "status": "failed",
                    "business_codes": ["E_CONFIG_INVALID"],
                    "failures": [failure],
                    "output_dir": str(output_dir),
                },
                sort_keys=True,
            )
        )
        return 2
    sequence_result = load_sequences(args.sequences)
    if not sequence_result.sequences:
        failures = sequence_result.failures or [
            public_failure(
                "E_SEQ_INVALID",
                "@file",
                "sequence file contains no valid sequence records",
            )
        ]
        _write_load_failure(output_dir, failures, mode="sequence", start_ns=start_ns)
        print(
            json.dumps(
                {
                    "status": "failed",
                    "business_codes": sorted(
                        {item["business_code"] for item in failures}
                    ),
                    "failures": failures,
                    "output_dir": str(output_dir),
                },
                sort_keys=True,
            )
        )
        return 2
    request = RunRequest(
        config=config,
        sequences=sequence_result.sequences,
        failures=sequence_result.failures,
        scale_tier=args.tier,
        state_dir=args.state_dir,
        output_dir=args.output_dir,
        fresh=args.fresh,
        expected_version=args.expected_version,
    )
    result = PipelineRunner(request).run()
    summary = {
        "status": "ok",
        "scale_tier": args.tier,
        "selected_strategy": result.decision_record["selected"],
        "sequence_count": len(result.predictions["sequences"]),
        "failure_count": len(result.failures),
        "failure_business_codes": result.diagnostic_report["failure_business_codes"],
        "stage_cache_totals": result.diagnostic_report["stage_cache_totals"],
        "snapshot_version": result.diagnostic_report["snapshot"]["global_version"],
        "snapshot_stale_codes": result.diagnostic_report["snapshot"]["stale_business_codes"],
        "elapsed_ms": result.diagnostic_report["monotonic_clock"]["elapsed_ms"],
        "output_files": result.output_files,
    }
    print(json.dumps(summary, sort_keys=True, ensure_ascii=True))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    config_result = load_config(args.config)
    sequence_result = load_sequences(args.sequences)
    payload = {
        "config_valid": config_result.ok,
        "config_failures": config_result.failures,
        "valid_sequence_count": len(sequence_result.sequences),
        "sequence_failures": sequence_result.failures,
        "failure_business_codes": sorted(
            {
                item["business_code"]
                for item in config_result.failures + sequence_result.failures
            }
        ),
    }
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0 if config_result.ok and not sequence_result.failures else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m motion_pipeline",
        description="Deterministic motion-time-series prediction pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="execute a full or incremental run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--sequences", required=True)
    run_parser.add_argument("--tier", type=int, required=True)
    run_parser.add_argument("--state-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--fresh", action="store_true")
    run_parser.add_argument("--expected-version", type=int, default=None)
    run_parser.set_defaults(func=cmd_run)

    validate_parser = subparsers.add_parser("validate", help="validate config and JSONL")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--sequences", required=True)
    validate_parser.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
