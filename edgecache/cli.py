"""Command line entry points for run, replay, diagnostics and spec checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .content import manifest_digest
from .dashboard import dashboard_snapshot, launch_ui
from .errors import EdgeError
from .replay import replay_manifest
from .scheduler import Coordinator
from .spec import load_manifest


def _emit(value: dict[str, Any], *, stream: Any = sys.stdout) -> None:
    json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
    stream.write("\n")
    stream.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgecache", description="Deterministic edge-cache shard lease dispatcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    spec_parser = subparsers.add_parser("spec-check", help="validate a declaration file")
    spec_parser.add_argument("--sample", required=True)

    run_parser = subparsers.add_parser("run", help="run the event stream against local node subprocesses")
    run_parser.add_argument("--sample", required=True)
    run_parser.add_argument("--db", required=True)
    run_parser.add_argument("--crash-node", default=None)
    run_parser.add_argument("--crash-after-grants", type=int, default=0)
    run_parser.add_argument("--retry-unknown", action="store_true")

    replay_parser = subparsers.add_parser("replay", help="independently replay and verify a ledger")
    replay_parser.add_argument("--sample", required=True)
    replay_parser.add_argument("--db", required=True)

    ui_parser = subparsers.add_parser("ui", help="open the read-only Tkinter diagnostic view")
    ui_parser.add_argument("--db", required=True)

    dashboard_parser = subparsers.add_parser("dashboard", help="print the stable read-only dashboard snapshot")
    dashboard_parser.add_argument("--db", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "spec-check":
            manifest = load_manifest(args.sample)
            _emit(
                {
                    "ok": True,
                    "format_version": manifest.format_version,
                    "manifest_digest": manifest_digest(manifest),
                    "nodes": [node.node_id for node in manifest.nodes],
                    "contents": [content.content_key for content in manifest.contents],
                    "event_count": len(manifest.events),
                }
            )
            return 0
        if args.command == "run":
            manifest = load_manifest(args.sample)
            coordinator = Coordinator(
                manifest,
                args.db,
                sample_path=Path(args.sample),
                crash_node=args.crash_node,
                crash_after_grants=args.crash_after_grants,
                retry_unknown=args.retry_unknown,
            )
            report = coordinator.run()
            _emit(report)
            return 0 if report.get("status") == "OK" else 3
        if args.command == "replay":
            manifest = load_manifest(args.sample)
            result = replay_manifest(args.db, manifest)
            _emit(result)
            return 0
        if args.command == "ui":
            launch_ui(args.db)
            return 0
        if args.command == "dashboard":
            _emit(dashboard_snapshot(args.db))
            return 0
        raise RuntimeError(f"unsupported command {args.command}")
    except EdgeError as exc:
        _emit({"ok": False, "error": exc.to_dict()}, stream=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
