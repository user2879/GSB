"""Command line entry points for local runs and recovery diagnostics."""
from __future__ import annotations

import argparse
import json
import os

from .config import load_config_dir
from .errors import ConfigError
from .ledger import Ledger
from .receipts import C3
from .runner import RunOrchestrator, full_rebuild
from .snapshot import SnapshotStore, build_diagnostic_report


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="edge-cache")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the validated command stream")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--work-dir", required=True)
    run_p.add_argument("--format", default=C3, choices=("c1", "c2", "c3"))

    rebuild_p = sub.add_parser("rebuild", help="clean-slate full rebuild")
    rebuild_p.add_argument("--config", required=True)
    rebuild_p.add_argument("--work-dir", required=True)
    rebuild_p.add_argument("--format", default=C3, choices=("c1", "c2", "c3"))

    validate_p = sub.add_parser("validate", help="validate a configuration batch")
    validate_p.add_argument("--config", required=True)

    recover_p = sub.add_parser("recover", help="inspect the last committed snapshot")
    recover_p.add_argument("--work-dir", required=True)

    args = parser.parse_args(argv)
    if args.command == "validate":
        try:
            config = load_config_dir(args.config)
        except ConfigError as exc:
            _print({"ok": False, "failures": [item.to_dict() for item in exc.failures]})
            return 2
        _print({"ok": True, "config_hash": config.source_hash,
                "nodes": len(config.nodes), "shards": len(config.shards),
                "quotas": len(config.quotas), "commands": len(config.commands)})
        return 0

    if args.command == "run":
        orchestrator = RunOrchestrator(args.config, args.work_dir, args.format)
        _print(orchestrator.run())
        return 0

    if args.command == "rebuild":
        _print(full_rebuild(args.config, args.work_dir, args.format))
        return 0

    snapshot_store = SnapshotStore(os.path.join(args.work_dir, "snapshots"))
    ledger = Ledger(os.path.join(args.work_dir, "ledger"))
    recovery = snapshot_store.load_current()
    report = build_diagnostic_report(
        recovery, ledger,
        format_counts={"c1": 0, "c2": 0, "c3": 0},
        node_advances={},
        budget_consumed=recovery.get("manifest", {}).get("cursor", {}).get("budget_consumed", 0),
    )
    _print(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
