from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .acceptance import run_acceptance
from .canonical import write_json_atomic
from .crash import run_crash_child
from .errors import ScaleReviewError
from .http_health import make_health_server
from .portcheck import check_loopback_port
from .propagation import VersionedPlan
from .replay import recover_and_replay
from .sample_data import write_sample_bundle
from .spec import load_spec_file
from .ledger import Ledger


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scale-review")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-sample")

    validate = sub.add_parser("validate")
    validate.add_argument("--spec", required=True)

    compute = sub.add_parser("compute")
    compute.add_argument("--spec", required=True)
    compute.add_argument("--out", default="artifacts/compute-report.json")
    compute.add_argument("--artifact-dir", default="artifacts")

    port = sub.add_parser("port-check")
    port.add_argument("--host", default="127.0.0.1")
    port.add_argument("--port", type=int, default=0)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    crash = sub.add_parser("crash-child")
    crash.add_argument("--spec", required=True)
    crash.add_argument("--db", required=True)
    crash.add_argument("--minute", type=int, default=4)

    recover = sub.add_parser("recover")
    recover.add_argument("--spec", required=True)
    recover.add_argument("--db", required=True)
    recover.add_argument("--out")

    acceptance = sub.add_parser("acceptance")
    acceptance.add_argument("--root", default=".")

    sub.add_parser("gui")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init-sample":
            paths = write_sample_bundle(Path.cwd())
            print_json({"ok": True, "paths": {k: str(v) for k, v in paths.items()}})
            return 0

        if args.command == "validate":
            spec = load_spec_file(args.spec)
            print_json(
                {
                    "ok": True,
                    "service_ids": spec.service_ids,
                    "minute_count": spec.minute_count,
                    "spec_digest": __import__(
                        "scale_review.canonical", fromlist=["sha256_hex"]
                    ).sha256_hex(spec.canonical_bytes()),
                }
            )
            return 0

        if args.command == "compute":
            spec = load_spec_file(args.spec)
            plan = VersionedPlan(spec)
            audit = plan.audit_algorithm_paths()
            report = {
                "ok": audit["equal"],
                "spec_path": args.spec,
                "plan_version": plan.version,
                "selection": plan.selection.ledger_basis(),
                "audit": audit,
                "digest": plan.digest(),
            }
            if args.out:
                write_json_atomic(args.out, report)
            print_json(report)
            return 0 if audit["equal"] else 3

        if args.command == "port-check":
            print_json(check_loopback_port(args.host, args.port))
            return 0

        if args.command == "serve":
            server = make_health_server(args.host, args.port)
            host, port = server.server_address
            print_json({"ok": True, "health_url": f"http://{host}:{port}/health"})
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
            return 0

        if args.command == "crash-child":
            run_crash_child(args.spec, args.db, minute=args.minute)
            return 0

        if args.command == "recover":
            spec = load_spec_file(args.spec)
            with Ledger(args.db) as ledger:
                result = recover_and_replay(ledger, spec)
            if args.out:
                write_json_atomic(args.out, result)
            print_json(
                {
                    "ok": True,
                    "recovered_attempts": len(result["recovered_attempts"]),
                    "replayed_rounds": len(result["replay"]["rounds"]),
                    "next_minute": result["next_minute"],
                    "out": args.out,
                }
            )
            return 0

        if args.command == "acceptance":
            report = run_acceptance(args.root)
            print_json({"ok": report["ok"], "report": report["report_path"]})
            return 0 if report["ok"] else 1

        if args.command == "gui":
            from .gui import main as gui_main

            return gui_main()

        parser.error(f"unknown command {args.command}")
        return 2
    except ScaleReviewError as exc:
        print_json(exc.to_dict())
        return {
            "E_SPEC": 2,
            "E_DIVERGE": 3,
            "E_LEASE": 4,
            "E_REPLAY": 5,
        }.get(exc.code, 1)
    except Exception as exc:  # pragma: no cover - CLI safety boundary
        print_json({"ok": False, "error": {"code": "E_INTERNAL", "message": str(exc)}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
