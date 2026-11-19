"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .canonical import hash_without
from .constants import ERROR_EXIT_CODES, PIPELINE_VERSION
from .errors import PipelineError
from .generator import generate_workspace
from .ledger import load_and_verify_ledger
from .runner import recover_workspace, run_workspace
from .sample import run_sample, sample_exit_code
from .verifier import compare_ledgers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rts-pipeline",
        description="Deterministic RTS resource/unit scheduling pipeline",
    )
    parser.add_argument("--version", action="version", version=PIPELINE_VERSION)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate rules, events, and truth ledger")
    generate.add_argument("--out", required=True)
    generate.add_argument("--seed", type=int, required=True)
    generate.add_argument("--ticks", type=int, default=36)
    generate.add_argument("--factions", type=int, default=4)
    generate.add_argument("--chunk-ticks", type=int, default=7)
    generate.add_argument(
        "--scenario",
        choices=("normal", "overflow", "contention"),
        default="normal",
    )
    generate.add_argument("--overflow-tick", type=int)
    generate.add_argument("--overflow-faction", default="B")

    run = subparsers.add_parser("run", help="synthesize chunks, verify, and commit reports")
    run.add_argument("--workspace", required=True)
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--memory-budget", type=int, default=8 * 1024 * 1024)
    run.add_argument("--chunk-ticks", type=int)

    sample = subparsers.add_parser("sample", help="generate and verify the complete sample corpus")
    sample.add_argument("--out", default="samples")
    sample.add_argument("--base-seed", type=int)

    verify = subparsers.add_parser("verify", help="verify a committed ledger and optionally compare")
    verify.add_argument("--ledger", required=True)
    verify.add_argument("--compare")

    recover = subparsers.add_parser("recover", help="recover only from a committed ledger")
    recover.add_argument("--workspace", required=True)
    return parser


def _print_json(value: Any) -> None:
    from .canonical import pretty_bytes

    sys.stdout.buffer.write(pretty_bytes(value))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            overflow_at = None
            overflow_tick = args.overflow_tick
            if args.scenario == "overflow" and overflow_tick is None:
                overflow_tick = max(0, args.ticks // 2)
            if overflow_tick is not None:
                overflow_at = (overflow_tick, args.overflow_faction)
            result = generate_workspace(
                Path(args.out),
                seed=args.seed,
                ticks=args.ticks,
                scenario=args.scenario,
                faction_count=args.factions,
                chunk_ticks=args.chunk_ticks,
                overflow_at=overflow_at,
            )
            _print_json(result["generation_report"])
            return 0
        if args.command == "run":
            result = run_workspace(
                Path(args.workspace),
                threads=args.threads,
                memory_budget_bytes=args.memory_budget,
                chunk_ticks=args.chunk_ticks,
            )
            view = {
                "status": result["status"],
                "result_code": result["result_code"],
                "exit_report_written": result["exit_report_written"],
            }
            if "exit_report" in result:
                view["exit_report"] = result["exit_report"]
            if "error" in result:
                view["error"] = result["error"]
            if "scheduler_report" in result and result["status"] != "ok":
                view["scheduler"] = result["scheduler_report"]
            _print_json(view)
            return int(result["exit_code"])
        if args.command == "sample":
            report = run_sample(Path(args.out), base_seed=args.base_seed)
            _print_json(
                {
                    "all_checks_passed": report["all_checks_passed"],
                    "sample_report_hash": report["sample_report_hash"],
                    "reproducibility": report["reproducibility"],
                    "checks": report["checks"],
                }
            )
            return sample_exit_code(report)
        if args.command == "verify":
            ledger = load_and_verify_ledger(Path(args.ledger))
            view: dict[str, Any] = {
                "valid": True,
                "ledger_hash": ledger["ledger_hash"],
                "business_hash": ledger["business_hash"],
                "summary": ledger["summary"],
                "result_codes": ledger["result_codes"],
            }
            if args.compare:
                other = load_and_verify_ledger(Path(args.compare))
                comparison = compare_ledgers(ledger, other)
                view["comparison"] = comparison
                view["result_code"] = comparison["result_code"]
            view["verification_hash"] = hash_without(view, "verification_hash")
            _print_json(view)
            return 0 if view.get("result_code", "OK") == "OK" else ERROR_EXIT_CODES[view["result_code"]]
        if args.command == "recover":
            report = recover_workspace(Path(args.workspace))
            _print_json(report)
            return 0
    except PipelineError as exc:
        _print_json(exc.to_dict())
        return exc.exit_code
    except Exception as exc:
        _print_json(
            {
                "code": "E_INTERNAL",
                "message": "unexpected CLI failure",
                "details": {"error_type": type(exc).__name__, "error": str(exc)},
            }
        )
        return ERROR_EXIT_CODES["E_INTERNAL"]
    return ERROR_EXIT_CODES["E_INTERNAL"]


if __name__ == "__main__":
    raise SystemExit(main())
