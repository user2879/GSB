from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .runner import run_directory
from .samplegen import SEED, write_sample


def _configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _run(args: argparse.Namespace) -> int:
    result = run_directory(
        args.input,
        args.output,
        drain=not args.no_drain,
        compare_full=not args.no_compare,
        reset=args.reset,
    )
    parity = result.get("parity") or {}
    injections = result.get("diagnostics", {}).get("injections", {})
    summary = {
        "status": result.get("status"),
        "output": str(Path(args.output)),
        "summary": result.get("summary"),
        "parity_pass": parity.get("all_required_parity_checks_pass"),
        "injections_isolated": injections.get("all_isolated"),
        "active_failures": len(result.get("active_failures", [])),
    }
    _print(summary)
    if result.get("status") != "OK":
        return 2
    if parity and not parity.get("all_required_parity_checks_pass", False):
        return 3
    if injections and not injections.get("all_isolated", True):
        return 3
    return 0


def _generate(args: argparse.Namespace) -> int:
    result = write_sample(args.output, seed=args.seed)
    _print(result)
    return 0


def _verify(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="mtcomp-verify-") as directory:
        root = Path(directory)
        sample = root / "sample"
        output = root / "out"
        generated = write_sample(sample, seed=SEED)
        first = run_directory(sample, output, reset=True, compare_full=True)
        first_rows = {
            row["id"]: row for row in first.get("sentences", [])
        }
        first_payload = {
            key: {
                field: row.get(field)
                for field in (
                    "translation",
                    "fragments",
                    "confidence_permille",
                    "confidence_fixed",
                    "status",
                    "boundary",
                    "boundary_detail",
                    "references",
                    "cache",
                )
            }
            for key, row in first_rows.items()
        }
        second = run_directory(sample, output, reset=False, compare_full=True)
        second_payload = {
            row["id"]: {
                field: row.get(field)
                for field in (
                    "translation",
                    "fragments",
                    "confidence_permille",
                    "confidence_fixed",
                    "status",
                    "boundary",
                    "boundary_detail",
                    "references",
                    "cache",
                )
            }
            for row in second.get("sentences", [])
        }
        resumed = bool(
            second.get("diagnostics", {})
            .get("snapshot", {})
            .get("resumed")
        )
        boundary_count = sum(
            1 for row in first.get("sentences", []) if row.get("boundary")
        )
        checks = {
            "sample_generated": len(generated["files"]) == 6,
            "first_run_ok": first.get("status") == "OK",
            "full_rebuild_parity": bool(
                (first.get("parity") or {}).get(
                    "all_required_parity_checks_pass"
                )
            ),
            "injections_isolated": bool(
                first.get("diagnostics", {})
                .get("injections", {})
                .get("all_isolated")
            ),
            "boundary_sample_present": boundary_count >= 1,
            "second_run_resumed": resumed,
            "resume_payload_identical": first_payload == second_payload,
        }
        report = {
            "seed": SEED,
            "checks": checks,
            "boundary_count": boundary_count,
            "checks_pass": all(checks.values()),
            "first_summary": first.get("summary"),
            "second_summary": second.get("summary"),
        }
        _print(report)
        return 0 if report["checks_pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mtcomp",
        description="Deterministic incremental sentence translation component",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run incremental translation")
    run_parser.add_argument("--input", required=True, help="sample input directory")
    run_parser.add_argument("--output", required=True, help="artifact directory")
    run_parser.add_argument("--no-drain", action="store_true")
    run_parser.add_argument("--no-compare", action="store_true")
    run_parser.add_argument("--reset", action="store_true")
    run_parser.set_defaults(handler=_run)

    generate_parser = subparsers.add_parser(
        "generate-sample", help="write the deterministic local sample"
    )
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--seed", type=int, default=SEED)
    generate_parser.set_defaults(handler=_generate)

    verify_parser = subparsers.add_parser(
        "verify", help="generate, run, compare and resume-check the sample"
    )
    verify_parser.set_defaults(handler=_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
