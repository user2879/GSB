from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import threading
from pathlib import Path
from typing import Any

from .samplegen import generate_sample
from .util import atomic_write_json
from .workbench import ConfigMismatchError, Workbench, WorkbenchError, error_payload


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "generate-samples":
        paths = generate_sample(args.out)
        print(json.dumps({name: str(path) for name, path in paths.items()}, ensure_ascii=False))
        return 0
    if args.command == "status":
        from .snapshot import SnapshotManager

        manager = SnapshotManager(args.snapshot)
        current = manager.load_current()
        if current is None:
            print(json.dumps({"status": "empty"}, ensure_ascii=False))
            return 0
        print(
            json.dumps(
                {
                    "status": current["manifest"].get("status"),
                    **current["manifest"],
                    "directory": current["directory"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate":
        from .validation import run_validation

        report = run_validation(
            samples_dir=args.samples,
            out_dir=args.out,
            cancellation_rounds=args.cancellation_rounds,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.get("all_checks_passed") else 1

    stop_event = threading.Event()

    def request_stop(signum: int, _frame: Any) -> None:
        stop_event.set()
        print(
            json.dumps(
                {
                    "event": "graceful_stop_requested",
                    "signal": signum,
                    "message": "finishing the current round before snapshotting",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if args.fresh:
        for path in (args.snapshot, args.out):
            target = Path(path)
            if target.exists():
                shutil.rmtree(target)

    try:
        workbench = Workbench(
            events_path=args.events,
            lexicon_path=args.lexicon,
            config_path=args.config,
            edits_path=args.edits,
            out_dir=args.out,
            snapshot_dir=args.snapshot,
        )
        result = workbench.run(
            resume=args.resume,
            stop_after_rounds=args.stop_after_rounds,
            crash_after_snapshots=args.simulate_kill_after_snapshots,
            stop_requested=stop_event.is_set,
        )
        if result.get("__crash_requested__"):
            sys.stderr.write("simulated forced termination\n")
            sys.stderr.flush()
            os._exit(137)
        summary = {
            "status": result.get("status"),
            "mode": result.get("mode"),
            "snapshot_version": result.get("recovery", {}).get("snapshot_version"),
            "cursor": result.get("recovery", {}).get("cursor"),
            "phase": result.get("recovery", {}).get("phase"),
            "rounds": len(result.get("summary_lengths", [])),
            "failures": len(result.get("failures", [])),
            "out": str(Path(args.out)),
            "snapshot_dir": str(Path(args.snapshot)),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except ConfigMismatchError as exc:
        payload = error_payload(exc.code, str(exc))
        Path(args.out).mkdir(parents=True, exist_ok=True)
        atomic_write_json(Path(args.out) / "error.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    except WorkbenchError as exc:
        payload = error_payload(exc.code, str(exc))
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    except Exception as exc:
        payload = error_payload(type(exc).__name__, str(exc))
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dialogwb",
        description="Deterministic incremental dialogue summary workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run baseline and edit batches")
    run.add_argument("--events", required=True)
    run.add_argument("--lexicon", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--edits")
    run.add_argument("--out", required=True)
    run.add_argument("--snapshot", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--fresh", action="store_true", help="remove out/snapshot first")
    run.add_argument("--stop-after-rounds", type=int)
    run.add_argument(
        "--simulate-kill-after-snapshots",
        type=int,
        help=argparse.SUPPRESS,
    )

    status = subparsers.add_parser("status", help="show the current snapshot manifest")
    status.add_argument("--snapshot", required=True)

    samples = subparsers.add_parser("generate-samples", help="generate fixed-seed inputs")
    samples.add_argument("--out", default="samples")

    validate = subparsers.add_parser("validate", help="run the full acceptance matrix")
    validate.add_argument("--samples", default="samples")
    validate.add_argument("--out", default="artifacts/validation")
    validate.add_argument("--cancellation-rounds", type=int, default=3)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

