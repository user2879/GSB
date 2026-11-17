"""Command-line entry points for routing, peers, snapshots, and replay."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from .config import load_config
from .engine import TunnelEngine
from .errors import TunnelError
from .peer import run_peer
from .replay import ReplayVerifier
from .snapshot import SnapshotStore, restore_snapshot


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tunnelvpn", description="Loopback tunnel lab")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Strictly validate JSON config")
    validate.add_argument("--config", required=True)

    run = sub.add_parser("run", help="Run all loopback listeners")
    run.add_argument("--config", required=True)
    run.add_argument("--state-dir", default="state")
    run.add_argument("--log", default=None)
    run.add_argument("--snapshot-on-stop", action="store_true")

    peer = sub.add_parser("peer", help="Run one simulated peer")
    peer.add_argument("--config", required=True)
    peer.add_argument("--peer-id", required=True)
    peer.add_argument("--log", default=None)

    snapshots = sub.add_parser("snapshots", help="Inspect snapshot storage")
    snapshots.add_argument("--state-dir", default="state")

    restore = sub.add_parser("restore", help="Restore latest snapshot as a whole")
    restore.add_argument("--config", required=True, help="Current config only for version comparison")
    restore.add_argument("--state-dir", default="state")
    restore.add_argument("--path", default=None)

    replay = sub.add_parser("replay", help="Strict deterministic snapshot replay")
    replay.add_argument("--config", required=True)
    replay.add_argument("--state-dir", default="state")
    replay.add_argument("--path", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            config = load_config(args.config)
            _print({"code": "OK", "config": config.to_dict()})
            return 0
        if args.command == "peer":
            config = load_config(args.config)
            run_peer(config, args.peer_id, args.log)
            return 0
        if args.command == "run":
            config = load_config(args.config)
            store = SnapshotStore(args.state_dir)
            engine = TunnelEngine(config, log_path=args.log)
            engine.start()
            _print(
                {
                    "code": "OK",
                    "message": "engine running; press Ctrl+C to stop",
                    "routes": [route.to_dict() for route in config.routes],
                }
            )
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
            finally:
                engine.stop()
                if args.snapshot_on_stop:
                    snapshot = store.save_engine(engine)
                    _print(
                        {
                            "code": "OK",
                            "snapshot": str(snapshot.path),
                            "ledger_digest": engine.ledger.digest(),
                        }
                    )
            return 0
        if args.command == "snapshots":
            store = SnapshotStore(args.state_dir)
            _print([entry.to_dict() for entry in store.list_snapshots()])
            return 0
        if args.command == "restore":
            config = load_config(args.config)
            store = SnapshotStore(args.state_dir)
            snapshot = store.load(args.path) if args.path else store.load_latest()
            restored = restore_snapshot(snapshot, config)
            _print(
                {
                    "code": "OK",
                    "report": restored.report.to_dict(),
                    "restored_route_version": restored.config.route_version,
                    "ledger_digest": restored.ledger.digest(),
                    "unknown_sessions": [
                        state.session_id for state in restored.sessions if state.requires_rerun
                    ],
                }
            )
            return 0
        if args.command == "replay":
            config = load_config(args.config)
            store = SnapshotStore(args.state_dir)
            snapshot = store.load(args.path) if args.path else store.load_latest()
            report = ReplayVerifier(snapshot, current_config=config).verify()
            _print(report.to_dict())
            return 0 if report.ok else 1
        raise AssertionError(args.command)
    except TunnelError as exc:
        _print(exc.to_dict())
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
