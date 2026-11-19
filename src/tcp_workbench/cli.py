"""Command-line entry point for validation, runs, snapshots, recovery, and UI."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Any

from .config import load_session_config
from .errors import E_PORT, E_SCRIPT, BusinessError
from .ledger import Ledger
from .peer import DeterministicPeer
from .runner import ingest_events, load_event_script, run_loopback_session
from .service import SessionService
from .snapshot import seed_snapshot
from .structured_log import StructuredLogger
from .ui_model import build_ui_model, render_ui_text

DEFAULT_CONFIG = "samples/session.json"
DEFAULT_DB = "var/tcpdiag.sqlite3"
DEFAULT_LOG = "logs/tcpdiag.jsonl"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _load(
    config_path: str,
    script_path: str | None = None,
    *,
    script_error_code: str = "E_CONFIG",
):
    return load_session_config(
        config_path,
        peer_script_override=script_path,
        script_error_code=script_error_code,
    )


def _open_service(
    args: argparse.Namespace,
    logger: StructuredLogger,
    *,
    reset: bool = False,
) -> SessionService:
    db_path = Path(args.db)
    if reset and db_path.exists():
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(db_path) + suffix)
            if sidecar.exists():
                sidecar.unlink()
    config, script = _load(args.config, getattr(args, "script", None))
    return SessionService.open(db_path, config, script, logger=logger)


def _print_result(value: Any) -> None:
    print(_json(value))


def command_validate(args: argparse.Namespace, logger: StructuredLogger) -> int:
    config, script = _load(args.config, args.script)
    _print_result(
        {
            "valid": True,
            "session_id": config.session_id,
            "config_version": config.config_version,
            "script_version": script.script_version,
            "config_digest": config.digest,
            "script_digest": script.digest,
            "script_responses": len(script.responses),
        }
    )
    logger.info(
        "config_validated",
        config_version=config.config_version,
        script_version=script.script_version,
    )
    return 0


def command_run(args: argparse.Namespace, logger: StructuredLogger) -> int:
    service = _open_service(args, logger, reset=args.reset)
    try:
        events = load_event_script(args.events)
        results = ingest_events(service, events)
        service.verify_incremental_equivalence()
        snapshot_id = (
            service.save_snapshot(args.snapshot_name) if args.snapshot_name else None
        )
        summary = {
            "session_id": service.config.session_id,
            "events_ingested": len(results),
            "last_event_seq": service.events[-1].event_seq if service.events else 0,
            "config_version": service.config.config_version,
            "script_version": service.script.script_version,
            "incremental_equals_full": True,
            "snapshot_id": snapshot_id,
            "conclusions": {
                key: {
                    "value": service.items[key].value,
                    "revision": service.items[key].revision,
                }
                for key in service.items
            },
        }
        _print_result(summary)
        logger.info("event_run_complete", events=len(results), snapshot_id=snapshot_id)
        return 0
    finally:
        service.close()


def command_loopback(args: argparse.Namespace, logger: StructuredLogger) -> int:
    service = _open_service(args, logger, reset=args.reset)
    try:
        transcript, peer = run_loopback_session(
            service,
            port=args.port,
            send_segments=args.segments,
            close_mode=args.close_mode,
        )
        service.verify_incremental_equivalence()
        snapshot_id = (
            service.save_snapshot(args.snapshot_name) if args.snapshot_name else None
        )
        _print_result(
            {
                "peer_address": f"127.0.0.1:{peer.port}",
                "rounds": transcript.rounds,
                "peer_responses": list(transcript.peer_responses),
                "dropped_seqs": list(transcript.dropped_seqs),
                "event_count": len(service.events),
                "incremental_equals_full": True,
                "snapshot_id": snapshot_id,
                "header": build_ui_model(service)["header"],
            }
        )
        logger.info(
            "loopback_run_complete",
            port=peer.port,
            events=len(service.events),
            snapshot_id=snapshot_id,
        )
        return 0
    finally:
        service.close()


def command_recover(args: argparse.Namespace, logger: StructuredLogger) -> int:
    service = SessionService.open_existing(args.db, logger=logger)
    try:
        service.verify_incremental_equivalence()
        report = service.recovery_report.as_dict() if service.recovery_report else {}
        _print_result(report)
        logger.info("recovery_complete", **report)
        return 0
    finally:
        service.close()


def command_retry(args: argparse.Namespace, logger: StructuredLogger) -> int:
    service = SessionService.open_existing(args.db, logger=logger)
    try:
        results = service.retry_unknown_events()
        service.verify_incremental_equivalence()
        _print_result(
            {
                "retried_event_seqs": [result.event.event_seq for result in results],
                "recovery": None
                if service.recovery_report is None
                else service.recovery_report.as_dict(),
            }
        )
        return 0
    finally:
        service.close()


def command_compare(args: argparse.Namespace, logger: StructuredLogger) -> int:
    service = SessionService.open_existing(args.db, logger=logger)
    try:
        service.verify_incremental_equivalence()
        print(
            _json({"incremental_equals_full": True, "event_count": len(service.events)})
        )
        logger.info("incremental_compare_passed", event_count=len(service.events))
        return 0
    finally:
        service.close()


def command_evolve(args: argparse.Namespace, logger: StructuredLogger) -> int:
    service = SessionService.open_existing(args.db, logger=logger)
    try:
        config, script = _load(args.config, args.script)
        report = service.evolve(config, script)
        service.verify_incremental_equivalence()
        _print_result(report.as_dict())
        return 0
    finally:
        service.close()


def command_ui(args: argparse.Namespace, logger: StructuredLogger) -> int:
    if args.reset:
        # UI normally opens an existing session; reset is an explicit demo aid.
        service = _open_service(args, logger, reset=True)
    else:
        try:
            service = SessionService.open_existing(args.db, logger=logger)
        except BusinessError as exc:
            if exc.code == "E_VERSION" and args.config:
                service = _open_service(args, logger)
            else:
                raise
    try:
        if getattr(args, "inject_missing_version", False):
            raise BusinessError(
                "E_VERSION",
                "snapshot version is missing (injected parameter)",
                ["snapshot.config_version"],
            )
        model = build_ui_model(service)
        if args.headless:
            print(render_ui_text(model))
            return 0
        try:
            from .ui import run_ui
        except ImportError as exc:
            logger.error(
                "ui_unavailable", code="E_UI", message=f"Tkinter is unavailable: {exc}"
            )
            print(f"E_UI: Tkinter is unavailable: {exc}", file=sys.stderr)
            return 3
        run_ui(service)
        return 0
    finally:
        service.close()


def command_snapshot(args: argparse.Namespace, logger: StructuredLogger) -> int:
    if args.snapshot_action == "save":
        service = SessionService.open_existing(args.db, logger=logger)
        try:
            snapshot_id = service.save_snapshot(args.name)
            _print_result({"snapshot_id": snapshot_id, "name": args.name})
            return 0
        finally:
            service.close()
    if args.snapshot_action == "list":
        ledger = Ledger(args.db)
        try:
            payload = [
                {
                    "snapshot_id": int(row["snapshot_id"]),
                    "name": row["name"],
                    "config_version": row["config_version"],
                    "script_version": row["script_version"],
                    "event_count": int(row["event_count"]),
                    "created_mono_ms": int(row["created_mono_ms"]),
                    "content_digest": row["content_digest"],
                }
                for row in ledger.snapshots()
            ]
            _print_result(payload)
            return 0
        finally:
            ledger.close()
    if args.snapshot_action == "verify":
        service = SessionService.open_existing(args.db, logger=logger)
        try:
            _print_result(service.verify_snapshot(args.snapshot_id))
            return 0
        finally:
            service.close()
    if args.snapshot_action == "restore":
        service = SessionService.open_existing(args.db, logger=logger)
        try:
            report = service.restore_snapshot(
                args.snapshot_id,
                inject_missing_version=args.inject_missing_version,
            )
            _print_result(report.as_dict())
            return 0
        finally:
            service.close()
    raise ValueError(f"unknown snapshot action: {args.snapshot_action}")


def command_seed_snapshots(args: argparse.Namespace, logger: StructuredLogger) -> int:
    ledger = Ledger(args.db)
    seeded = []
    try:
        for path in sorted(Path(args.directory).glob("*.json")):
            seed = json.loads(path.read_text(encoding="utf-8"))
            snapshot_id = seed_snapshot(
                ledger,
                seed,
                mono_ms=int(seed.get("created_mono_ms", 0)),
                base_dir=path.parent,
            )
            seeded.append(
                {
                    "file": str(path),
                    "snapshot_id": snapshot_id,
                    "name": seed.get("name"),
                }
            )
        _print_result(seeded)
        logger.info("snapshots_seeded", count=len(seeded))
        return 0
    finally:
        ledger.close()


def command_inject(args: argparse.Namespace, logger: StructuredLogger) -> int:
    if args.kind == "script":
        _load(args.config, args.script, script_error_code=E_SCRIPT)
        raise BusinessError(
            E_SCRIPT, "script corruption injection did not trigger", ["peer_script"]
        )
    if args.kind == "port":
        _config, script = _load(args.config, args.script)
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            blocker.bind(("127.0.0.1", args.port))
            blocker.listen(1)
            peer = DeterministicPeer(script, "127.0.0.1", args.port)
            try:
                peer.start()
                raise BusinessError(
                    E_PORT, "port occupancy injection did not trigger", ["port"]
                )
            finally:
                peer.stop()
        finally:
            blocker.close()
    if args.kind == "version":
        service = SessionService.open_existing(args.db, logger=logger)
        try:
            service.restore_snapshot(args.snapshot_id, inject_missing_version=True)
            raise BusinessError(
                "E_VERSION",
                "version injection did not trigger",
                ["snapshot.config_version"],
            )
        finally:
            service.close()
    raise ValueError(f"unknown injection kind: {args.kind}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tcpdiag", description="TCP state and congestion diagnostics workbench"
    )
    parser.add_argument(
        "--log", default=DEFAULT_LOG, help="structured JSON-lines log path"
    )
    parser.add_argument(
        "--log-echo", action="store_true", help="also print structured log records"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate session config and peer script"
    )
    validate.add_argument("--config", default=DEFAULT_CONFIG)
    validate.add_argument("--script")
    validate.set_defaults(func=command_validate)

    run = subparsers.add_parser("run", help="ingest a deterministic event sequence")
    run.add_argument("--db", default=DEFAULT_DB)
    run.add_argument("--config", default=DEFAULT_CONFIG)
    run.add_argument("--script")
    run.add_argument("--events", required=True)
    run.add_argument("--snapshot-name")
    run.add_argument("--reset", action="store_true")
    run.set_defaults(func=command_run)

    loopback = subparsers.add_parser(
        "loopback", help="run a real 127.0.0.1 deterministic peer session"
    )
    loopback.add_argument("--db", default=DEFAULT_DB)
    loopback.add_argument("--config", default=DEFAULT_CONFIG)
    loopback.add_argument("--script")
    loopback.add_argument("--port", type=int, default=0)
    loopback.add_argument("--segments", type=int, default=6)
    loopback.add_argument(
        "--close-mode", choices=("close_wait", "fin_ack"), default="close_wait"
    )
    loopback.add_argument("--snapshot-name")
    loopback.add_argument("--reset", action="store_true")
    loopback.set_defaults(func=command_loopback)

    recover = subparsers.add_parser(
        "recover", help="recover from the durable event ledger"
    )
    recover.add_argument("--db", default=DEFAULT_DB)
    recover.set_defaults(func=command_recover)

    retry = subparsers.add_parser(
        "retry", help="retry unknown complete-payload events from the ledger"
    )
    retry.add_argument("--db", default=DEFAULT_DB)
    retry.set_defaults(func=command_retry)

    compare = subparsers.add_parser(
        "compare", help="compare incremental conclusions with full replay"
    )
    compare.add_argument("--db", default=DEFAULT_DB)
    compare.set_defaults(func=command_compare)

    evolve = subparsers.add_parser(
        "evolve", help="replace conclusions for a new config/script version"
    )
    evolve.add_argument("--db", default=DEFAULT_DB)
    evolve.add_argument("--config", required=True)
    evolve.add_argument("--script")
    evolve.set_defaults(func=command_evolve)

    ui = subparsers.add_parser("ui", help="open the Tkinter workbench")
    ui.add_argument("--db", default=DEFAULT_DB)
    ui.add_argument("--config")
    ui.add_argument("--script")
    ui.add_argument(
        "--headless", action="store_true", help="render the stable UI model as text"
    )
    ui.add_argument("--reset", action="store_true")
    ui.add_argument("--inject-missing-version", action="store_true")
    ui.set_defaults(func=command_ui)

    snapshot = subparsers.add_parser("snapshot", help="manage snapshots")
    snapshot_sub = snapshot.add_subparsers(dest="snapshot_action", required=True)
    snapshot_save = snapshot_sub.add_parser("save")
    snapshot_save.add_argument("--db", default=DEFAULT_DB)
    snapshot_save.add_argument("--name", required=True)
    snapshot_list = snapshot_sub.add_parser("list")
    snapshot_list.add_argument("--db", default=DEFAULT_DB)
    snapshot_verify = snapshot_sub.add_parser("verify")
    snapshot_verify.add_argument("--db", default=DEFAULT_DB)
    snapshot_verify.add_argument("--snapshot-id", type=int, required=True)
    snapshot_restore = snapshot_sub.add_parser("restore")
    snapshot_restore.add_argument("--db", default=DEFAULT_DB)
    snapshot_restore.add_argument("--snapshot-id", type=int, required=True)
    snapshot_restore.add_argument("--inject-missing-version", action="store_true")
    snapshot.set_defaults(func=command_snapshot)

    seed = subparsers.add_parser(
        "seed-snapshots", help="seed historical snapshot samples into SQLite"
    )
    seed.add_argument("--db", default=DEFAULT_DB)
    seed.add_argument("--directory", default="samples/snapshots")
    seed.set_defaults(func=command_seed_snapshots)

    inject = subparsers.add_parser(
        "inject", help="inject E_SCRIPT, E_PORT, or E_VERSION failures"
    )
    inject_sub = inject.add_subparsers(dest="kind", required=True)
    inject_script = inject_sub.add_parser("script")
    inject_script.add_argument("--config", default=DEFAULT_CONFIG)
    inject_script.add_argument("--script", required=True)
    inject_port = inject_sub.add_parser("port")
    inject_port.add_argument("--config", default=DEFAULT_CONFIG)
    inject_port.add_argument("--script")
    inject_port.add_argument("--port", type=int, required=True)
    inject_version = inject_sub.add_parser("version")
    inject_version.add_argument("--db", default=DEFAULT_DB)
    inject_version.add_argument("--snapshot-id", type=int, required=True)
    inject.set_defaults(func=command_inject)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = StructuredLogger(args.log, echo=args.log_echo)
    command = args.command + (
        f":{getattr(args, 'kind', '')}" if getattr(args, "kind", "") else ""
    )
    logger.info("command_started", command=command)
    try:
        result = args.func(args, logger)
        logger.info("command_succeeded", command=command)
        return int(result)
    except BusinessError as exc:
        logger.business_error("command_failed", exc, command=command)
        print(_json(exc.as_dict()), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "command_crashed", code="E_INTERNAL", message=str(exc), command=command
        )
        print(
            _json(
                {"code": "E_INTERNAL", "message": str(exc), "fields": [], "context": {}}
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
