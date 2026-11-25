"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import load_bundle
from .engine import DiagnosticEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Categorical cross encoding budget diagnostics"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one diagnostic build")
    run_parser.add_argument("--features", required=True, type=Path)
    run_parser.add_argument("--strategy", type=Path, default=None)
    run_parser.add_argument("--state-dir", type=Path, default=Path(".diagnostic_state"))
    run_parser.add_argument("--report", type=Path, required=True)
    run_parser.add_argument("--serial", action="store_true")
    run_parser.add_argument("--install-signal-handlers", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="validate YAML and exit")
    validate_parser.add_argument("--features", required=True, type=Path)
    validate_parser.add_argument("--strategy", type=Path, default=None)
    return parser


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        bundle = load_bundle(args.features, args.strategy)
        payload = {
            "ok": bundle.ok,
            "issue_count": len(bundle.issues),
            "issues": bundle.issues,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if bundle.ok else 2

    if args.command == "run":
        bundle = load_bundle(args.features, args.strategy)
        engine = DiagnosticEngine(
            bundle, state_dir=args.state_dir, serial=bool(args.serial)
        )
        if args.install_signal_handlers:
            engine.install_signal_handlers()
        report = engine.run()
        write_json_atomic(args.report, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "ok": report["ok"],
                    "failure_count": report["failure_count"],
                    "combination_count": report["combination_count"],
                    "adjudicated_count": report["adjudicated_count"],
                    "pending_count": len(report["pending"]),
                    "report": str(args.report),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0 if report["ok"] else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
