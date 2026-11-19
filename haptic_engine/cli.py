"""Command line entry point for the haptic synthesis engine."""

from __future__ import annotations

import argparse
import json
import sys

from .engine import HapticEngine, SignalController
from .scheduler import InterruptRequested
from .errors import E_DIGEST, E_SYNC, HapticError
from .parser import load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthesize per-device haptic command streams.")
    parser.add_argument("manifest", help="Path to the deterministic input JSON manifest")
    parser.add_argument("--serial", action="store_true", help="Force concurrency=1 reference path")
    parser.add_argument(
        "--block-delay",
        type=float,
        default=0.0,
        help="Optional per-block delay in seconds, useful for signal/recovery tests",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except HapticError as exc:
        print(exc.to_json(), file=sys.stderr)
        return 2

    signals = SignalController()
    signals.install()
    engine = HapticEngine(
        manifest,
        serial=args.serial,
        block_delay=args.block_delay,
        signals=signals,
    )
    try:
        report = engine.run()
    except InterruptRequested:
        exit_code = engine.handle_interrupt_and_exit()
        print(
            json.dumps(
                {
                    "ok": False,
                    "interrupted": True,
                    "exit_code": exit_code,
                    "state_dir": manifest.state_dir,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return exit_code
    except HapticError as exc:
        print(exc.to_json(), file=sys.stderr)
        if exc.code == E_SYNC:
            return 3
        if exc.code == E_DIGEST:
            return 4
        return 5

    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    if not report["ok"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
