"""Command-line interface for the workbench."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from .errors import WorkbenchError
from .pipeline import CheckpointHook, run_workbench


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kg-workbench",
        description=(
            "Deterministic Chinese KG triple extraction with fixed-point "
            "confidence fusion and two-precision diagnostics."
        ),
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="path to the extraction config JSON (default: config.json)",
    )
    parser.add_argument("--snapshot-dir", help="override paths.snapshot_dir")
    parser.add_argument("--report", help="override paths.report")
    parser.add_argument("--workers", type=int, help="override runtime.workers")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="discard the current snapshot and start from the load stage",
    )
    parser.add_argument(
        "--stop-after-stage",
        choices=("load", "entity", "relation", "fusion", "diagnosis"),
        help="test hook: commit the named stage and exit with status 0",
    )
    return parser


def _summary(result: dict[str, Any], report_path: Path) -> dict[str, Any]:
    if result.get("status") == "interrupted":
        return result
    return {
        "status": result["status"],
        "report": str(report_path),
        "snapshot_version": result["snapshot"]["version"],
        "triples": len(result["triples"]),
        "diagnostics": len(result["diagnostics"]),
        "boundary_sensitive_triples": len(result["boundary_sensitive_triples"]),
        "injection_hits": sum(
            1 for item in result["injection_hits"] if item.get("hit")
        ),
        "failures": len(result["failures"]),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    stop_event = threading.Event()

    def handle_signal(signum: int, _frame: Any) -> None:
        stop_event.set()

    for signal_name in ("SIGINT", "SIGTERM"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is not None:
            try:
                signal.signal(signal_value, handle_signal)
            except (ValueError, OSError):
                pass

    hooks = CheckpointHook(
        stop_event=stop_event,
        stop_after_stage=args.stop_after_stage,
    )
    try:
        result = run_workbench(
            args.config,
            snapshot_dir=args.snapshot_dir,
            report_path=args.report,
            workers=args.workers,
            fresh=args.fresh,
            hooks=hooks,
        )
    except WorkbenchError as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return exc.exit_code
    except KeyboardInterrupt:
        stop_event.set()
        print(
            json.dumps(
                {
                    "status": "interrupted",
                    "exit_code": 0,
                    "message": "signal received; committed snapshots remain resumable",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    report_path = Path(args.report).resolve() if args.report else None
    if report_path is None:
        from .loader import load_config

        report_path = load_config(args.config).report_path
    print(
        json.dumps(
            _summary(result, Path(report_path)),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.get("status") in {"completed", "interrupted"} else 5


if __name__ == "__main__":
    raise SystemExit(main())
