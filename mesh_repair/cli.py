"""Command-line entry points for the deterministic mesh repair pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .errors import EXIT_CODES, BusinessError, ErrorCode
from .hashing import canonical_json_bytes, write_json
from .pipeline import run_fixed_pipeline
from .verification import validate_files


def _print_json(value: object) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value, pretty=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mesh_repair",
        description="Deterministic 3D mesh defect generation, repair and verification",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the fixed end-to-end sequence")
    run_parser.add_argument("--config", required=True, help="JSON configuration file")
    run_parser.add_argument("--out", required=True, help="output directory")
    run_parser.add_argument("--threads", type=int, default=None, help="override thread cap")
    run_parser.add_argument(
        "--memory-budget-bytes", type=int, default=None, help="override memory budget"
    )
    run_parser.add_argument(
        "--block-time-budget-ms", type=float, default=None, help="override per-block time budget"
    )

    validate_parser = subparsers.add_parser(
        "validate", help="verify a final PLY using only the PLY and ground-truth JSON"
    )
    validate_parser.add_argument("--mesh", required=True)
    validate_parser.add_argument("--truth", required=True)
    validate_parser.add_argument("--out", default=None, help="optional verifier report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_fixed_pipeline(
                args.config,
                args.out,
                threads=args.threads,
                memory_budget_bytes=args.memory_budget_bytes,
                block_time_budget_ms=args.block_time_budget_ms,
            )
            summary = {
                "status": result.status,
                "business_code": result.business_code,
                "exit_code": result.exit_code,
                "step_hashes": result.hashes["steps"],
                "report_sha256": result.hashes["report.json"]["sha256"],
            }
            if result.report.get("error"):
                summary["error"] = result.report["error"]
            _print_json(summary)
            return result.exit_code

        if args.command == "validate":
            report = validate_files(args.mesh, args.truth)
            if args.out:
                write_json(args.out, report)
            _print_json(report)
            if report["status"] == "pass":
                return 0
            code = ErrorCode(report.get("business_code", ErrorCode.E_RESIDUAL_TOPOLOGY.value))
            return EXIT_CODES.get(code, 1)
        parser.error("unknown command")
    except BusinessError as exc:
        _print_json(exc.to_dict())
        return EXIT_CODES.get(exc.code, 1)
    except Exception as exc:  # pragma: no cover - process boundary
        _print_json(
            {
                "status": "fail",
                "business_code": ErrorCode.E_INTERNAL.value,
                "message": "unhandled command exception",
                "details": {"error": repr(exc)},
                "exit_code": EXIT_CODES[ErrorCode.E_INTERNAL],
            }
        )
        return EXIT_CODES[ErrorCode.E_INTERNAL]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
