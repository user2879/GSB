"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
from pathlib import Path

from .engine import INTERRUPTED_EXIT_CODE, SignalEvent, run_engine
from .errors import EngineError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitle-engine",
        description="Deterministic incremental subtitle layout and accessibility renderer",
    )
    parser.add_argument("--input", required=True, type=Path, help="input JSON description")
    parser.add_argument("--state-dir", type=Path, help="override versioned state directory")
    parser.add_argument("--output-dir", type=Path, help="override atomic output directory")
    parser.add_argument("--concurrency", type=int, help="override runtime.concurrency")
    parser.add_argument("--memory-budget-bytes", type=int, help="override memory budget")
    parser.add_argument(
        "--block-delay-ms",
        type=int,
        default=0,
        help="diagnostic per-block computation delay for signal tests",
    )
    parser.add_argument(
        "--clean-state",
        action="store_true",
        help="remove the entire state directory before starting",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "run":
        args_list = args_list[1:]
    args = _parser().parse_args(args_list)

    if args.clean_state:
        state_path = args.state_dir.resolve() if args.state_dir else None
        if state_path is None:
            # Input-relative default is resolved by the parser; read only enough
            # to honor the same path here would duplicate validation.  Require
            # an explicit path for this destructive convenience flag.
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": {
                            "code": "E_PATH",
                            "message": "--clean-state requires an explicit --state-dir",
                            "details": {},
                        },
                    }
                )
            )
            return 2
        if state_path.exists():
            shutil.rmtree(state_path)

    stop_event = SignalEvent()

    def _handle(signum: int, _frame: object) -> None:
        stop_event.request(signum)

    previous_int = signal.signal(signal.SIGINT, _handle)
    previous_term = signal.signal(signal.SIGTERM, _handle)
    try:
        result = run_engine(
            args.input,
            state_dir=args.state_dir,
            output_dir=args.output_dir,
            concurrency=args.concurrency,
            memory_budget_bytes=args.memory_budget_bytes,
            test_block_delay_ms=args.block_delay_ms,
            stop_event=stop_event,
        )
        print(json.dumps(result.summary(), ensure_ascii=False, indent=2))
        return result.exit_code
    except EngineError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2))
        return 2
    except KeyboardInterrupt:
        stop_event.request(2)
        print(
            json.dumps(
                {
                    "ok": False,
                    "status": "interrupted",
                    "exit_code": INTERRUPTED_EXIT_CODE,
                    "message": "KeyboardInterrupt accepted by signal handler boundary",
                }
            )
        )
        return INTERRUPTED_EXIT_CODE
    except Exception as exc:  # unexpected failures still return a business JSON shape
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "E_INTERNAL",
                        "message": str(exc),
                        "details": {"type": type(exc).__name__},
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
