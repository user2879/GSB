"""Command-line entry points for generation, validation, and diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import load_registration_config
from .engine import run_diagnostics
from .fingerprint import read_json, sha256_json
from .scenario import run_acceptance_scenario
from .synth import generate_demo


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _cmd_generate(args: argparse.Namespace) -> int:
    metadata = generate_demo(
        Path(args.out),
        seed=args.seed,
        positive_clips=args.positive_clips,
        negative_clips=args.negative_clips,
    )
    _print_json(metadata)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    failures: list = []
    config = load_registration_config(Path(args.config), failures)
    payload = {
        "valid": config is not None,
        "failure_count": len(failures),
        "failures": [failure.as_dict() for failure in failures],
    }
    _print_json(payload)
    return 0 if config is not None else 2


def _cmd_run(args: argparse.Namespace) -> int:
    result, report = run_diagnostics(
        Path(args.config),
        Path(args.state),
        result_path=Path(args.result),
        report_path=Path(args.report),
        mode=args.mode,
    )
    _print_json(
        {
            "failure_count": len(result["failures"]),
            "failure_codes": sorted({item["code"] for item in result["failures"]}),
            "conclusion_count": len(result["conclusions"]),
            "rejected_count": sum(1 for item in result["conclusions"] if item["rejected"]),
            "cluster_count": len(result["clusters"]["clusters"]),
            "confusion_candidate_count": len(result["confusion_candidates"]),
            "matrix_consistent": result["confusion_matrix"].get("consistency", {}).get("consistent", False),
            "pending_count": len(result["pending_queue"]),
            "result": str(Path(args.result)),
            "report": str(Path(args.report)),
            "state_version": report.get("snapshot", {}).get("state_version") if report.get("snapshot") else None,
            "cache": report["cache"],
        }
    )
    return 2 if result["failures"] else 0


def _cmd_compare(args: argparse.Namespace) -> int:
    incremental = read_json(Path(args.incremental))
    full = read_json(Path(args.full))
    equal = incremental == full
    payload = {
        "exactly_equal": equal,
        "incremental_sha256": sha256_json(incremental),
        "full_sha256": sha256_json(full),
    }
    _print_json(payload)
    return 0 if equal else 1


def _cmd_scenario(args: argparse.Namespace) -> int:
    summary = run_acceptance_scenario(Path(args.out), seed=args.seed)
    _print_json(summary)
    return 0 if summary["canonical_results_exactly_equal"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-diag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="generate deterministic local WAV corpus")
    generate.add_argument("--out", required=True)
    generate.add_argument("--seed", type=int, default=20261126)
    generate.add_argument("--positive-clips", type=int, default=4)
    generate.add_argument("--negative-clips", type=int, default=4)
    generate.set_defaults(func=_cmd_generate)

    validate = sub.add_parser("validate", help="validate registration config")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=_cmd_validate)

    run = sub.add_parser("run", help="run diagnostics")
    run.add_argument("--config", required=True)
    run.add_argument("--state", required=True)
    run.add_argument("--result", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--mode", choices=("auto", "full"), default="auto")
    run.set_defaults(func=_cmd_run)

    compare = sub.add_parser("compare", help="compare two canonical result files")
    compare.add_argument("--incremental", required=True)
    compare.add_argument("--full", required=True)
    compare.set_defaults(func=_cmd_compare)

    scenario = sub.add_parser("scenario", help="run the complete incremental/full acceptance scenario")
    scenario.add_argument("--out", required=True)
    scenario.add_argument("--seed", type=int, default=20261126)
    scenario.set_defaults(func=_cmd_scenario)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
