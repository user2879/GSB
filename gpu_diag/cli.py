"""Command line interface for the GPU quota & preemption diagnostic workbench."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import DiagError, StructuredLogger
from .persist.ledger import Ledger
from .samplegen import inject_fault, write_samples
from .session import (authoritative_recompute, recover_session,
                      reset_state_dir, restore_from_snapshot, run_session)


def _print(payload) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))


def _paths(args) -> tuple[Path, Path, Path]:
    base = Path(args.samples)
    return base / "pools.json", base / "tasks.json", base / "params.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-diag",
        description="GPU quota & shard preemption scheduling diagnostic workbench")
    parser.add_argument("--samples", default="samples",
                        help="directory holding pools.json/tasks.json/params.json")
    parser.add_argument("--state", default="state", help="ledger/snapshot directory")
    parser.add_argument("--log", default="logs/events.jsonl",
                        help="structured JSON-lines log path")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-samples", help="write deterministic sample JSON files")

    run = sub.add_parser("run", help="run a full deterministic scheduling session")
    run.add_argument("--fail-shard", action="append", default=[],
                     help="shard global id to fail (repeatable), e.g. T2#1")
    run.add_argument("--crash-after-round", type=int, default=None,
                     help="simulate kill -9 after this round is dispatched")

    sub.add_parser("recover", help="reopen ledger and reconcile unknown shards")
    sub.add_parser("recompute", help="authoritative replay vs ledger (E_PREVIEW)")
    sub.add_parser("restore", help="snapshot whole-state restore + E_REPLAY check")
    sub.add_parser("verify-ledger", help="verify ledger digest chain (E_LEDGER)")

    inject = sub.add_parser("inject", help="inject one of four business faults")
    inject.add_argument("fault", choices=["memory", "quota", "cycle", "ledger"])
    sub.add_parser("ui", help="launch the Tkinter diagnostic window")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = StructuredLogger(args.log if args.command != "init-samples" else None)

    if args.command == "init-samples":
        write_samples(args.samples)
        _print({"written": sorted(p.name for p in Path(args.samples).iterdir())})
        return 0

    pools_path, tasks_path, params_path = _paths(args)

    try:
        if args.command == "inject":
            reset_state_dir(args.state)
            write_samples(args.samples)
            if args.fault == "ledger":
                result = run_session(pools_path, tasks_path, params_path,
                                     args.state, logger)
                ledger = Ledger(Path(args.state) / "ledger.sqlite", 1,
                                result.declaration_digest)
                ledger.tamper_round_water(1)
                ledger.close()
                ledger2 = Ledger(Path(args.state) / "ledger.sqlite", 1,
                                 result.declaration_digest)
                ledger2.verify_digests()
                ledger2.close()
            else:
                inject_fault(args.samples, args.fault)
                run_session(pools_path, tasks_path, params_path, args.state, logger)
            return 0

        if args.command == "run":
            result = run_session(
                pools_path, tasks_path, params_path, args.state, logger,
                fail_shards=set(args.fail_shard) or None,
                crash_after_round=args.crash_after_round)
            _print({"rounds": len(result.rounds),
                    "unknown_shards": result.unknown_shards,
                    "outcomes": result.outcomes,
                    "declaration_digest": result.declaration_digest})
            return 0

        if args.command == "recover":
            _print(recover_session(pools_path, tasks_path, params_path,
                                   args.state, logger))
            return 0

        if args.command == "recompute":
            _print(authoritative_recompute(pools_path, tasks_path, params_path,
                                           args.state))
            return 0

        if args.command == "restore":
            _print(restore_from_snapshot(pools_path, tasks_path, params_path,
                                         args.state))
            return 0

        if args.command == "verify-ledger":
            from .session import verify_ledger_digest
            verify_ledger_digest(args.state)
            _print({"ledger": "digest chain verified"})
            return 0

        if args.command == "ui":
            from .ui.app import launch_ui
            launch_ui(args.state)
            return 0
    except DiagError as exc:
        record = logger.log(exc.code, exc.message, details=exc.details)
        _print(record)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
