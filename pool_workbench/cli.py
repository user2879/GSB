"""Command line entry point for the connection-pool workbench.

Examples
--------
python -m pool_workbench.cli build-db
python -m pool_workbench.cli run --config samples/tasks_cold.json
python -m pool_workbench.cli snapshot
python -m pool_workbench.cli replay --snapshot snapshots/snapshot-v1-001.json
python -m pool_workbench.cli inject --kind schema_missing_table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .app import WorkbenchSession
from .errors import BusinessError
from .fault_injection import (
    inject_config_quota_exceed,
    inject_schema_missing_table,
    inject_snapshot_missing_version,
    inject_replay_data_version,
)
from .sample_db import INVENTORY_TABLE, ALLOWED_TABLES, build_sample_db


def _default_workspace() -> Path:
    return Path(__file__).resolve().parent.parent / "workspace"


def _print(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _handle(business_error: BusinessError) -> int:
    _print({"business_error": business_error.to_dict()})
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SQLite connection-pool workbench")
    parser.add_argument("--workspace", default=str(_default_workspace()))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-db", help="build the deterministic sample database")

    run = sub.add_parser("run", help="load a task script and execute the batch")
    run.add_argument("--config", required=True)
    run.add_argument("--save-snapshot", action="store_true")

    sub.add_parser("snapshot", help="save a snapshot of the latest run")

    snap = sub.add_parser("list-snapshots", help="list stored snapshots")

    restore = sub.add_parser("restore", help="restore a snapshot (whole replacement)")
    restore.add_argument("--snapshot", required=True)

    replay = sub.add_parser("replay", help="replay and validate a snapshot")
    replay.add_argument("--snapshot", required=True)

    inject = sub.add_parser("inject", help="run a startup-parameter fault injection")
    inject.add_argument(
        "--kind",
        required=True,
        choices=[
            "schema_missing_table",
            "config_quota_exceed",
            "snapshot_missing_version",
            "replay_data_version",
        ],
    )
    inject.add_argument("--config", default=None)
    inject.add_argument("--snapshot", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace)
    db_path = workspace / "sample.sqlite"
    config_path = Path(args.config) if getattr(args, "config", None) else None

    if args.command == "build-db":
        meta = build_sample_db(db_path)
        _print({"built": str(db_path), **meta})
        return 0

    session = WorkbenchSession(workspace)
    try:
        if args.command == "run":
            session.load(config_path, db_path)
            result = session.execute()
            payload = result.to_dict()
            if args.save_snapshot:
                payload["snapshot"] = str(session.save_snapshot())
            _print(payload)
            session.close()
            return 0

        if args.command == "snapshot":
            if config_path is None:
                raise BusinessError("E_CONFIG", "snapshot requires --config", ["--config"])
            session.load(config_path, db_path)
            session.execute()
            _print({"snapshot": str(session.save_snapshot())})
            session.close()
            return 0

        if args.command == "list-snapshots":
            _print({"snapshots": [str(p) for p in session.store.list_snapshots()]})
            return 0

        if args.command == "restore":
            # Restore uses the snapshot itself; config is not required.
            restored = session.restore_snapshot(args.snapshot)
            _print({
                "snapshot_id": restored["snapshot_id"],
                "replaced": restored["replaced"],
                "diff": restored["diff"],
            })
            return 0

        if args.command == "replay":
            document = session.store.load(args.snapshot)
            config_for_version = workspace / "replay_session.json"
            if not config_for_version.exists():
                config_for_version = Path(args.snapshot)
            session.load_from_snapshot_for_replay(document, db_path)
            report = session.replay(document)
            _print(report)
            return 0

        if args.command == "inject":
            return _run_injection(args, session, workspace, db_path)

    except BusinessError as exc:
        return _handle(exc)
    return 0


def _run_injection(args, session: WorkbenchSession, workspace: Path,
                   db_path: Path) -> int:
    if args.kind == "schema_missing_table":
        build_sample_db(db_path)
        inject_schema_missing_table(db_path, INVENTORY_TABLE)
        config = Path(__file__).resolve().parent / "samples" / "tasks_cold.json"
        try:
            session.load(config, db_path)
        except BusinessError as exc:
            return _handle(exc)

    if args.kind == "config_quota_exceed":
        source = Path(args.config) if args.config else (
            Path(__file__).resolve().parent / "samples" / "tasks_cold.json"
        )
        injected = inject_config_quota_exceed(source)
        build_sample_db(db_path)
        try:
            session.load(injected, db_path)
        except BusinessError as exc:
            return _handle(exc)

    if args.kind == "snapshot_missing_version":
        snapshot = Path(args.snapshot) if args.snapshot else _latest_snapshot(session)
        injected = inject_snapshot_missing_version(snapshot)
        try:
            session.store.load(injected)
        except BusinessError as exc:
            return _handle(exc)

    if args.kind == "replay_data_version":
        snapshot = Path(args.snapshot) if args.snapshot else _latest_snapshot(session)
        document = session.store.load(snapshot)
        session.load_from_snapshot_for_replay(document, db_path)
        inject_replay_data_version(db_path)
        try:
            session.replay(document)
        except BusinessError as exc:
            return _handle(exc)

    _print({"warning": "injection did not raise the expected business error"})
    return 1


def _latest_snapshot(session: WorkbenchSession) -> Path:
    snapshots = session.store.list_snapshots()
    if not snapshots:
        raise BusinessError("E_SNAPSHOT", "no snapshot available", ["snapshots/"])
    return snapshots[-1]


if __name__ == "__main__":
    sys.exit(main())
