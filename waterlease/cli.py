from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .ledger import Ledger
from .process_manager import ProcessManager
from .scheduler import Scheduler
from .spec import load_spec
from .topology import topological_order


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), flush=True)


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec_dir)
    print_json(
        {
            "ok": True,
            "spec_digest": spec.spec_digest,
            "topological_order": topological_order(spec.shards),
            "node_count": len(spec.nodes),
            "shard_count": len(spec.shards),
        }
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec_dir)
    ledger = Ledger(args.ledger, spec)
    recovery = ledger.recover() if args.recover else None
    try:
        with ProcessManager(spec.nodes) as processes:
            scheduler = Scheduler(
                spec,
                ledger,
                processes,
                crash_after_inflight=args.crash_after_inflight,
            )
            report = scheduler.run()
        print_json({"ok": True, "recovery": recovery, "run": _json_safe(report)})
        return 0
    finally:
        ledger.close()


def cmd_recover(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec_dir)
    ledger = Ledger(args.ledger, spec)
    try:
        report = ledger.recover()
        print_json({"ok": True, "recovery": _json_safe(report)})
        return 0
    finally:
        ledger.close()


def cmd_replay(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec_dir)
    ledger = Ledger(args.ledger, spec)
    try:
        report = ledger.assert_replay_ok()
        print_json({"ok": True, "replay": report})
        return 0
    finally:
        ledger.close()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="waterlease")
    parser.add_argument("--spec-dir", default="spec")
    parser.add_argument("--ledger", default="ledger/waterlease.sqlite3")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run")
    run.add_argument("--recover", action="store_true")
    run.add_argument("--crash-after-inflight", type=int, default=0)
    run.set_defaults(func=cmd_run)

    recover = sub.add_parser("recover")
    recover.set_defaults(func=cmd_recover)

    replay = sub.add_parser("replay")
    replay.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
