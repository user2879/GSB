"""Command-line interface for the offline backup engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .engine import BackupEngine
from .samples import generate_samples
from .storage import atomic_write_json


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True, help="engine workspace directory")
    parser.add_argument(
        "--hmac-key-file",
        help="file containing the HMAC-SHA256 key bytes",
    )
    parser.add_argument(
        "--algorithms",
        default="sha256,hmac-sha256",
        help="comma-separated whitelist (default: sha256,hmac-sha256)",
    )
    parser.add_argument(
        "--report",
        help="write the JSON report atomically to this path; '-' means stdout",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backup_engine",
        description="Offline deterministic backup integrity and recovery engine",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser(
        "import", help="validate and import one or more local snapshot JSON files"
    )
    _add_common(import_parser)
    import_parser.add_argument("snapshots", nargs="+", metavar="SNAPSHOT")

    recover_parser = subparsers.add_parser(
        "recover", help="import optional snapshots and execute deterministic recovery"
    )
    _add_common(recover_parser)
    recover_parser.add_argument("snapshots", nargs="*", metavar="SNAPSHOT")

    status_parser = subparsers.add_parser("status", help="query the committed state")
    _add_common(status_parser)

    verify_parser = subparsers.add_parser(
        "verify-checkpoint",
        aliases=["validate-checkpoint", "checkpoint"],
        help="validate checkpoint and immutable restore generation integrity",
    )
    _add_common(verify_parser)

    samples_parser = subparsers.add_parser(
        "samples", help="generate deterministic local sample snapshots"
    )
    samples_parser.add_argument("--output", required=True, help="sample output directory")
    samples_parser.add_argument("--report", help="write JSON report atomically to this path")
    return parser


def _read_algorithms(parser: argparse.ArgumentParser, value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        parser.error("--algorithms must contain at least one algorithm")
    return result


def _read_hmac_key(parser: argparse.ArgumentParser, value: str | None) -> bytes | None:
    if value is None:
        return None
    try:
        return Path(value).expanduser().read_bytes()
    except OSError as exc:
        parser.error(f"cannot read --hmac-key-file: {exc}")
        raise AssertionError("unreachable")


def _emit_report(report: dict[str, Any], report_path: str | None) -> None:
    if report_path is None or report_path == "-":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return
    path = Path(report_path).expanduser()
    try:
        atomic_write_json(path, report)
    except OSError as exc:
        raise SystemExit(f"cannot write report {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "samples":
        report = generate_samples(args.output)
        _emit_report(report, args.report)
        return 0

    algorithms = _read_algorithms(parser, args.algorithms)
    hmac_key = _read_hmac_key(parser, args.hmac_key_file)
    try:
        engine = BackupEngine(
            args.workspace,
            hmac_key=hmac_key,
            algorithms=algorithms,
        )
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError("unreachable")

    if args.command == "import":
        report = engine.import_snapshots(args.snapshots)
    elif args.command == "recover":
        report = engine.recover(args.snapshots)
    elif args.command == "status":
        report = engine.status()
    elif args.command in {"verify-checkpoint", "validate-checkpoint", "checkpoint"}:
        report = engine.verify_checkpoint()
    else:  # pragma: no cover - argparse restricts the command
        parser.error(f"unsupported command: {args.command}")
        raise AssertionError("unreachable")

    _emit_report(report, args.report)
    return int(report["exit_code"])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
