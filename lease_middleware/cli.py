"""Command line interface for deterministic lease middleware tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .canonical import canonical_text
from .crash import crash_victim, run_crash_demo
from .errors import LeaseMiddlewareError
from .faults import run_injection
from .ledger import Ledger, recover_snapshot
from .service import run_demo
from .spec import load_sample_dir
from .verify import verify_ledger


def _print(payload: Any) -> None:
    print(canonical_text(payload), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leasectl")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate JSON specifications")
    validate.add_argument("--sample-dir", default="samples")

    demo = sub.add_parser("demo", help="run repeatable TCP and ledger demo")
    demo.add_argument("--root", default=".")

    verify = sub.add_parser("verify", help="replay and compare SQLite ledger")
    verify.add_argument("--sample-dir", default="samples")
    verify.add_argument("--ledger", required=True)

    inject = sub.add_parser("inject", help="inject a deterministic local fault")
    inject.add_argument(
        "--kind",
        required=True,
        choices=[
            "spec",
            "spec-json",
            "lease",
            "snapshot",
            "verify",
            "port",
            "timeout",
            "reply",
        ],
    )
    inject.add_argument("--root", default=".")

    crash = sub.add_parser("crash-demo", help="force-kill and recover")
    crash.add_argument("--root", default=".")

    victim = sub.add_parser("crash-victim", help=argparse.SUPPRESS)
    victim.add_argument("--dir", required=True)
    victim.add_argument("--port", type=int, default=48191)

    recover = sub.add_parser("recover-snapshot", help="replace invalid snapshot")
    recover.add_argument("--ledger", required=True)
    recover.add_argument("--snapshot", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            bundle = load_sample_dir(args.sample_dir)
            _print(
                {
                    "ok": True,
                    "code": "E_OK",
                    "nodes": len(bundle.nodes),
                    "requests": len(bundle.requests),
                    "events": len(bundle.events),
                    "global_in_flight": bundle.nodes_doc["quotas"]["global_in_flight"],
                }
            )
            return 0

        if args.command == "demo":
            report = run_demo(Path(args.root))
            comparison = report["comparison"]
            _print(
                {
                    "ok": True,
                    "code": "E_OK",
                    "byte_identical": comparison["byte_identical"],
                    "conclusion_sha256": comparison["forward_sha256"],
                    "artifacts": str(Path(args.root) / "artifacts" / "demo"),
                }
            )
            return 0

        if args.command == "verify":
            bundle = load_sample_dir(args.sample_dir)
            ledger = Ledger.open_readonly(args.ledger)
            try:
                report = verify_ledger(bundle, ledger)
            finally:
                ledger.close()
            _print(
                {
                    "ok": True,
                    "code": "E_OK",
                    "verified_records": report["verified_records"],
                    "ledger_digest": report["ledger_digest"],
                    "conclusion_digest": report["conclusion_digest"],
                    "summary": report["summary"],
                }
            )
            return 0

        if args.command == "inject":
            result = run_injection(args.kind, Path(args.root))
            _print(result)
            return 0 if result.get("expected_code") else 1

        if args.command == "crash-demo":
            report = run_crash_demo(Path(args.root))
            _print(
                {
                    "ok": True,
                    "code": "E_OK",
                    "victim_returncode": report["victim_returncode"],
                    "unknown_lease_count": len(report["recovery"]["unknown_leases"]),
                    "unknown_leases": report["recovery"]["unknown_leases"],
                    "snapshot_replaced": report["snapshot_recovery"]["replaced"],
                }
            )
            return 0

        if args.command == "crash-victim":
            crash_victim(args.dir, port=args.port)
            return 0

        if args.command == "recover-snapshot":
            ledger = Ledger.open(args.ledger)
            try:
                _, replacement = recover_snapshot(args.snapshot, ledger)
            finally:
                ledger.close()
            _print({"ok": True, "code": "E_OK", "snapshot_recovery": replacement})
            return 0

        raise ValueError(f"unknown command {args.command}")

    except LeaseMiddlewareError as exc:
        _print(exc.to_dict())
        return 2
    except Exception as exc:
        _print(
            {
                "ok": False,
                "code": "E_INTERNAL",
                "message": str(exc),
                "type": type(exc).__name__,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
