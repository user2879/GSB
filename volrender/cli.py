from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from .errors import PipelineError
from .pipeline import render_task
from .samples import generate_samples, verify_samples

VALIDATION_CODES = {
    "E_JSON",
    "E_TYPE",
    "E_SCHEMA",
    "E_DIMENSIONS",
    "E_SPACING",
    "E_VOLUME",
    "E_TRANSFER",
    "E_CAMERA",
    "E_OUTPUT",
    "E_BLOCK",
    "E_CONCURRENCY",
    "E_MEMORY",
    "E_EDIT",
}


def _print_json(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    sys.stdout.flush()


def _install_signal_handlers(event: threading.Event) -> None:
    def handler(_signum: int, _frame: Any) -> None:
        event.set()

    for name in ("SIGINT", "SIGTERM"):
        value = getattr(signal, name, None)
        if value is not None:
            try:
                signal.signal(value, handler)
            except (OSError, ValueError):
                pass


def _render_command(args: argparse.Namespace, interrupt_event: threading.Event) -> int:
    result = render_task(
        args.task,
        state_dir_override=args.state_dir,
        output_dir_override=args.output_dir,
        force_full=args.full,
        interrupt_event=interrupt_event,
    )
    _print_json(
        {
            "ok": True,
            "run_id": result.report["run_id"],
            "outputs": result.report["outputs"],
            "summary": result.report["summary"],
            "version_id": result.report["version"]["version_id"],
            "changes": result.report["changes"],
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="volrender",
        description="Deterministic offline volume renderer with geometric cache invalidation",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-samples", help="generate deterministic local samples")
    generate.add_argument("--samples", default="samples", help="sample directory")
    generate.add_argument("--seed", type=int, default=20261119, help="fixed random seed")

    render = subparsers.add_parser("render", help="render one JSON task")
    render.add_argument("task", help="task JSON file")
    render.add_argument("--state-dir", default=None, help="override state directory")
    render.add_argument("--output-dir", default=None, help="override output directory")
    render.add_argument("--full", action="store_true", help="bypass cache reuse for a full render")

    verify = subparsers.add_parser(
        "verify-samples", help="verify incremental samples against full renders"
    )
    verify.add_argument("--samples", default="samples", help="sample directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    interrupt_event = threading.Event()
    _install_signal_handlers(interrupt_event)

    try:
        if args.command == "generate-samples":
            manifest = generate_samples(args.samples, args.seed)
            _print_json({"ok": True, "samples": str(Path(args.samples).resolve()), **manifest})
            return 0
        if args.command == "render":
            return _render_command(args, interrupt_event)
        if args.command == "verify-samples":
            result = verify_samples(args.samples)
            _print_json(result)
            return 0 if result["ok"] else 1
        parser.error(f"unknown command: {args.command}")
        return 2
    except PipelineError as exc:
        _print_json(exc.to_dict())
        if exc.code == "E_INTERRUPTED":
            return 130
        if exc.code in VALIDATION_CODES:
            return 2
        return 1
    except (OSError, ValueError, KeyError) as exc:
        _print_json(
            {
                "ok": False,
                "code": "E_IO",
                "message": str(exc),
                "path": "$",
            }
        )
        return 1
    except Exception as exc:  # pragma: no cover - last-resort CLI guard
        _print_json(
            {
                "ok": False,
                "code": "E_INTERNAL",
                "message": f"{type(exc).__name__}: {exc}",
                "path": "$",
            }
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

