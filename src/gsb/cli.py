"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .demo import run_demo
from .errors import GSBError
from .ledger import Ledger
from .runtime import ClusterRuntime
from .spec import load_spec


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gsb-demo", description="GSB fixed-point shard diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="strictly validate a spec")
    validate.add_argument("spec")

    demo = subparsers.add_parser("demo", help="run the complete local acceptance demo")
    demo.add_argument("--spec", default="examples/spec.json")
    demo.add_argument("--evolved-spec", default="examples/spec_evolved.json")
    demo.add_argument("--workdir", default="artifacts/demo")

    run = subparsers.add_parser("run", help="execute every shard against real child nodes")
    run.add_argument("--spec", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--session-version", type=int, default=1)

    recover = subparsers.add_parser("recover", help="recover committed state after forcible termination")
    recover.add_argument("--ledger", required=True)
    recover.add_argument("--session-version", type=int, default=1)

    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            spec = load_spec(args.spec)
            _print({"valid": True, "topological_order": list(spec.topological_order()), "spec": spec.as_dict()})
            return 0
        if args.command == "demo":
            report = run_demo(args.spec, args.evolved_spec, args.workdir)
            _print(
                {
                    "status": report["status"],
                    "report": str(Path(args.workdir) / "report.json"),
                    "invalidation": report["invalidation"]["invalid_shards"],
                    "preserved": report["invalidation"]["preserved_shards"],
                    "error_codes": sorted(report["error_coverage"]),
                }
            )
            return 0
        if args.command == "run":
            spec = load_spec(args.spec)
            with Ledger(args.ledger, session_version=args.session_version) as ledger:
                recovery = ledger.recover()
                with ClusterRuntime(spec, ledger=ledger, session_version=args.session_version) as runtime:
                    result = runtime.execute()
                _print(
                    {
                        "status": "PASS",
                        "recovery": recovery,
                        "records": list(result.records),
                        "waits": result.waits,
                        "lease_stats": result.lease_stats,
                    }
                )
            return 0
        if args.command == "recover":
            with Ledger(args.ledger, session_version=args.session_version) as ledger:
                _print(ledger.recover())
            return 0
    except GSBError as exc:
        _print({"status": "FAIL", "error": exc.as_dict()})
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
