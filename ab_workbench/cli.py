"""Command line entry points for the evaluation workbench."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from typing import Any, Dict, List, Optional

from .config import load_config
from .loader import load_injections
from .pipeline import GracefulStop, Workbench


def _write_report(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, ensure_ascii=False, indent=2)


def _build_workbench(args, concurrent: bool) -> Workbench:
    config = load_config(args.config)
    return Workbench(
        config=config,
        snapshot_dir=args.snapshot_dir,
        cache_dir=args.cache_dir,
        tmp_dir=args.tmp_dir,
        concurrent=concurrent,
    )


def cmd_run(args) -> int:
    concurrent = not args.serial
    workbench = _build_workbench(args, concurrent)
    injections = load_injections(args.injections) if args.injections else []
    graceful = GracefulStop()
    signal.signal(signal.SIGTERM, graceful.handler)
    signal.signal(signal.SIGINT, graceful.handler)
    result = workbench.run(
        log_path=args.log,
        attributes_path=args.attributes,
        injections=injections,
        graceful=graceful,
    )
    payload = result.to_dict()
    _write_report(args.report, payload)
    print(f"report written to {args.report} (exit_code={result.exit_code}, mode={payload['mode']})")
    if result.signal_handled:
        print(f"graceful stop handled: {result.signal_handled}")
    return result.exit_code


def cmd_serial(args) -> int:
    args.serial = True
    return cmd_run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A/B offline evaluation workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--log", required=True)
        p.add_argument("--config", required=True)
        p.add_argument("--attributes", required=True)
        p.add_argument("--injections")
        p.add_argument("--snapshot-dir", required=True)
        p.add_argument("--cache-dir", required=True)
        p.add_argument("--tmp-dir", required=True)
        p.add_argument("--report", required=True)

    run_p = sub.add_parser("run", help="concurrent (or --serial) evaluation")
    run_p.add_argument("--serial", action="store_true")
    add_common(run_p)
    run_p.set_defaults(func=cmd_run)

    serial_p = sub.add_parser("serial", help="forced single-thread evaluation")
    add_common(serial_p)
    serial_p.set_defaults(func=cmd_serial)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
