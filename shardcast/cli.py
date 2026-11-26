"""Command line entry points for repeatable local demos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import Engine
from .ledger import Ledger
from .recovery import recover_and_replay
from .spec import load_spec


def _json_default(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def run_command(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    ledger = Ledger(args.ledger)
    engine = Engine(
        spec,
        ledger,
        socket_timeout=args.timeout,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
    )
    result = engine.run()
    payload = {
        "capacity_version": result.capacity_version,
        "outcomes": result.outcomes,
        "decisions": result.decisions,
        "capacity_diffs": result.capacity_diffs,
        "waits": result.waits,
        "water_marks": result.water_marks,
        "snapshot_replacement": getattr(engine, "snapshot_replacement", None),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    ledger.close()
    print(f"wrote {output_path}")
    return 0


def recover_command(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    report = recover_and_replay(
        spec,
        args.ledger,
        killed_node=args.killed_node,
        batch_size=args.batch_size,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="shardcast deterministic shard dispatcher")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a declared dispatch session")
    run.add_argument("--spec", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--timeout", type=float, default=3.0)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--max-attempts", type=int, default=3)
    run.set_defaults(func=run_command)

    recover = sub.add_parser("recover", help="recover after a forced worker kill")
    recover.add_argument("--spec", required=True)
    recover.add_argument("--ledger", required=True)
    recover.add_argument("--output", required=True)
    recover.add_argument("--killed-node")
    recover.add_argument("--batch-size", type=int, default=64)
    recover.set_defaults(func=recover_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
