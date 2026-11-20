from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import BusinessError
from .loader import load_events, load_parameter_snapshot, load_reconfigurations
from .logging import StructuredLogger
from .loopback import LoopbackSignalBroker, NullSignalBroker, check_loopback_port
from .replay import verify_replay
from .runner import ExperimentRunner, RunnerConfig
from .storage import SQLiteLedger


def _print_json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2))


def _add_common_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pools", required=True, help="pool declaration JSON")
    parser.add_argument("--borrows", required=True, help="borrow relation JSON")
    parser.add_argument("--trace", required=True, help="trace NDJSON")
    parser.add_argument("--log", default=None, help="structured JSONL log path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gateway_quota.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate all local inputs without processing")
    _add_common_inputs(validate)

    run = subparsers.add_parser("run", help="process trace in complete batches")
    _add_common_inputs(run)
    run.add_argument("--ledger", required=True, help="SQLite ledger path")
    run.add_argument("--reconfig", action="append", default=[], help="parameter reconfiguration JSON")
    run.add_argument("--batch-size", type=int, default=5)
    run.add_argument("--no-loopback", action="store_true", help="disable loopback signal handshakes")
    run.add_argument("--loopback-port", type=int, default=0, help="127.0.0.1 port; 0 selects a free port")
    run.add_argument("--crash-after-seq", type=int, default=None, help="force os._exit(137) before batch commit")

    replay = subparsers.add_parser("replay", help="replay committed event audit and compare with ledger")
    _add_common_inputs(replay)
    replay.add_argument("--ledger", required=True)

    return parser


def _load_inputs(args: argparse.Namespace) -> tuple[Any, list[Any], list[dict[str, Any]]]:
    snapshot = load_parameter_snapshot(args.pools, args.borrows)
    events = load_events(args.trace, snapshot)
    reconfigs = load_reconfigurations(args.reconfig) if getattr(args, "reconfig", None) else []
    return snapshot, events, reconfigs


def _command_validate(args: argparse.Namespace) -> int:
    snapshot, events, reconfigs = _load_inputs(args)
    _print_json(
        {
            "ok": True,
            "code": "OK",
            "parameter_version": snapshot.version,
            "pools": len(snapshot.pools),
            "borrow_rules": len(snapshot.borrow_rules),
            "events": len(events),
            "reconfigurations": len(reconfigs),
            "event_range": [events[0].seq, events[-1].seq] if events else None,
        }
    )
    return 0


def _command_run(args: argparse.Namespace) -> int:
    snapshot, events, reconfigs = _load_inputs(args)
    logger = StructuredLogger(args.log)
    if args.no_loopback:
        broker: Any = NullSignalBroker()
        transport = {"transport": "disabled", "socket_opened": False}
    else:
        selected_port = check_loopback_port(args.loopback_port)
        broker = LoopbackSignalBroker(selected_port).start()
        transport = {"transport": "loopback_tcp", "host": "127.0.0.1", "port": broker.port}
        logger.emit("LOOPBACK_READY", "loopback port is bound and free", **transport)
    try:
        outcome = ExperimentRunner(
            RunnerConfig(
                ledger_path=args.ledger,
                initial_snapshot=snapshot,
                events=events,
                reconfigurations=reconfigs,
                batch_size=args.batch_size,
                crash_after_seq=args.crash_after_seq,
            ),
            signal_broker=broker,
            logger=logger,
        ).run()
        outcome["transport"] = transport
        outcome["ledger_path"] = str(Path(args.ledger).resolve())
        _print_json(outcome)
        return 0
    finally:
        if hasattr(broker, "stop"):
            broker.stop()


def _command_replay(args: argparse.Namespace) -> int:
    snapshot, _, _ = _load_inputs(args)
    with SQLiteLedger(args.ledger) as store:
        store.initialize(snapshot)
        report = verify_replay(store)
        report["ledger_path"] = str(Path(args.ledger).resolve())
        _print_json(report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = StructuredLogger(getattr(args, "log", None), stream=sys.stderr)
    try:
        if args.command == "validate":
            return _command_validate(args)
        if args.command == "run":
            return _command_run(args)
        if args.command == "replay":
            return _command_replay(args)
        parser.error(f"unknown command: {args.command}")
    except BusinessError as exc:
        logger.validation_failure(exc)
        _print_json(exc.to_dict())
        return 2
    except Exception as exc:
        logger.emit("E_INTERNAL", str(exc), exception_type=type(exc).__name__)
        _print_json({"ok": False, "code": "E_INTERNAL", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
