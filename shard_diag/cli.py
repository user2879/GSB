from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .generator import generate_workspace
from .spec import load_spec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shard-diag", description="local shard-route and lease diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="write deterministic configs and samples")
    generate.add_argument("--root", default=".", help="workspace root")

    validate = subparsers.add_parser("validate", help="validate a local JSON specification")
    validate.add_argument("spec", help="path to specification JSON")

    demo = subparsers.add_parser("demo", help="run the full loopback/child/SQLite acceptance demo")
    demo.add_argument("--root", default=".", help="workspace root")
    demo.add_argument("--report", default=None, help="override report JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "generate":
        result = generate_workspace(args.root)
    elif args.command == "validate":
        try:
            spec = load_spec(args.spec)
        except Exception as error:
            payload = error.to_dict() if hasattr(error, "to_dict") else {"ok": False, "message": str(error)}
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 2
        result = {
            "ok": True,
            "spec": str(Path(args.spec).resolve()),
            "databases": [database.id for database in spec.databases],
            "table_versions": spec.table_versions(),
            "events": len(spec.events),
            "spec_digest": spec.spec_digest(),
            "event_digest": spec.event_digest(),
        }
    elif args.command == "demo":
        from .demo import run_demo

        result = run_demo(args.root, report_path=args.report)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())

