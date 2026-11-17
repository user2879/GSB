"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .canonical import atomic_write_json
from .engine import PipelineEngine
from .state import SnapshotStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feature-pipeline",
        description="Incremental deterministic feature and label-delayed sample pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run or resume one staged pipeline pass")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--events", type=Path, required=True)
    run_parser.add_argument("--features", type=Path, required=True)
    run_parser.add_argument("--labels", type=Path, required=True)
    run_parser.add_argument("--edits", type=Path)
    run_parser.add_argument("--injections", type=Path)
    run_parser.add_argument("--state-dir", type=Path, default=Path("runtime"))
    run_parser.add_argument("--report", type=Path)
    run_parser.add_argument("--reset", action="store_true", help="discard prior snapshots")
    run_parser.add_argument(
        "--stop-after",
        choices=("events", "features", "samples"),
        help="test hook: stop gracefully after the selected atomic stage commit",
    )

    status_parser = subparsers.add_parser(
        "status", help="inspect the latest complete snapshot without mutating it"
    )
    status_parser.add_argument("--state-dir", type=Path, default=Path("runtime"))
    status_parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        store = SnapshotStore(args.state_dir)
        state, recovery = store.recover()
        payload = {
            "recovery_point": recovery,
            "global_version": state.global_version,
            "completed_stages": state.completed_stages,
            "event_cursor": state.event_cursor,
            "event_count": len(state.events),
            "feature_count": len(state.features),
            "sample_count": len(state.samples),
            "snapshot_versions": state.manifest.get("stage_versions", {}),
        }
        _emit(payload, args.report)
        return 0

    engine = PipelineEngine(args.state_dir)
    try:
        report, exit_code = engine.run(
            config_path=args.config,
            events_path=args.events,
            features_path=args.features,
            labels_path=args.labels,
            edits_path=args.edits,
            injections_path=args.injections,
            reset=args.reset,
            stop_after=args.stop_after,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary must return an integer
        payload = {
            "fatal_error": type(exc).__name__,
            "message": str(exc),
        }
        _emit(payload, args.report)
        return 1
    _emit(report, args.report)
    return exit_code


def _emit(payload: dict[str, Any], report_path: Path | None) -> None:
    if report_path is not None:
        atomic_write_json(report_path, payload)
    summary = {
        key: payload.get(key)
        for key in (
            "global_version",
            "completed_stages",
            "event_cursor",
            "feature_count",
            "sample_count",
            "empty_window_feature_count",
            "stopped_gracefully",
        )
        if key in payload
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())

