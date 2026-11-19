"""Command-line interface for demo generation and VAD execution."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from .config import ConfigError
from .demo import generate_demo
from .pipeline import run_vad


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vad.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate-demo", help="generate deterministic local WAV samples and manifest"
    )
    generate.add_argument("--data-dir", default="demo/input")
    generate.add_argument("--config", default="demo/vad_config.json")
    generate.add_argument("--manifest", default="demo/sample_manifest.json")
    generate.add_argument("--seed", type=int, default=20261120)

    run = subparsers.add_parser("run", help="run VAD over an input directory")
    run.add_argument("--input", default="demo/input")
    run.add_argument("--config", default="demo/vad_config.json")
    run.add_argument("--manifest", default="demo/sample_manifest.json")
    run.add_argument("--state-dir", default=".vad_state")
    run.add_argument("--output", default="reports/latest.json")
    run.add_argument("--serial", action="store_true", help="force one worker")
    run.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="reduce concurrency, never above configuration",
    )
    run.add_argument(
        "--strict",
        action="store_true",
        help="return exit code 4 when sample failures or error diagnostics exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate-demo":
        try:
            manifest = generate_demo(
                args.data_dir,
                args.config,
                args.manifest,
                seed=args.seed,
            )
        except (ConfigError, OSError, ValueError) as exc:
            print(f"E_CONFIG: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "status": "generated",
                    "samples": len(manifest["samples"]),
                    "data_dir": str(Path(args.data_dir)),
                    "manifest": str(Path(args.manifest)),
                    "seed": manifest["generator"]["seed"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    stop_event = threading.Event()
    previous_handlers: dict[int, Any] = {}

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            previous_handlers[signum] = signal.signal(signum, request_stop)
        except (OSError, ValueError):
            pass
    try:
        report = run_vad(
            args.input,
            args.config,
            args.manifest,
            args.state_dir,
            args.output,
            concurrency_override=1 if args.serial else args.concurrency,
            stop_event=stop_event,
        )
    except ConfigError as exc:
        print(f"E_CONFIG: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"E_IO: {exc}", file=sys.stderr)
        return 2
    finally:
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass

    print(
        json.dumps(
            {
                "status": report["status"],
                "global_version": report["global_version"],
                "samples_processed": report["summary"]["samples_processed"],
                "endpoint_segments": report["summary"]["endpoint_segments"],
                "failures": report["summary"]["failed_samples"],
                "cache_reuse": report["cache_audit"]["reuse_count"],
                "cache_recompute": report["cache_audit"]["recompute_count"],
                "result_digest": report["result_digest"],
                "report": str(Path(args.output)),
                "resume_cursor": report["recovery"]["resume_cursor"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if args.strict and (
        report["failures"]
        or any(item.get("severity") == "error" for item in report["diagnostics"])
    ):
        return 4
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
