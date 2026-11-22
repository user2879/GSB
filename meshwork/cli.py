from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

from .engine import MeshRuntime
from .errors import MeshError
from .ledger import Ledger
from .logging_utils import StructuredLogger
from .model import MeshSpec, RequestSpec, load_requests, load_spec, parse_requests
from .replay import compare_ledgers
from .snapshot import SnapshotStore
from .upstream_manager import UpstreamManager


def _default_session_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%dT%H%M%S', time.localtime())}-{os.getpid()}"


def _logger(path: str | None) -> StructuredLogger:
    return StructuredLogger(path)


def _print_json(value: Any, stream: Any = sys.stdout) -> None:
    json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()


def _load_runtime_spec(args: argparse.Namespace, logger: StructuredLogger) -> MeshSpec:
    if args.inject == "corrupt-rules":
        source = Path(args.spec)
        raw = json.loads(source.read_text(encoding="utf-8"))
        injected_dir = Path("artifacts") / "injected"
        injected_dir.mkdir(parents=True, exist_ok=True)
        target = injected_dir / f"corrupt-rules-{os.getpid()}-{int(time.time())}.json"
        revisions = raw.get("route_revisions")
        if isinstance(revisions, list) and len(revisions) > 1 and isinstance(revisions[1], dict):
            revisions[1]["revision"] = revisions[0].get("revision")
        else:
            raw["route_revisions"] = [{"revision": 0, "rules": []}]
        target.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.warning(
            "anomaly_injection",
            code="E_SPEC",
            kind="corrupt-rules",
            source=args.spec,
            injected_file=str(target),
        )
        return load_spec(target)
    return load_spec(args.spec)


def _spec_revision(spec: MeshSpec, requested: int | None) -> Any:
    return spec.active_revision(requested)


def _reserve_injected_port(spec: MeshSpec, logger: StructuredLogger) -> socket.socket:
    instance = spec.ordered_services()[0].instances[0]
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name == "nt":
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            holder.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    else:
        holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind((instance.host, instance.port))
    holder.listen(1)
    logger.warning(
        "anomaly_injection",
        code="E_PORT",
        kind="port-occupied",
        instance_id=instance.id,
        host=instance.host,
        port=instance.port,
    )
    return holder


def _corrupt_latest_snapshot(
    store: SnapshotStore,
    runtime: MeshRuntime | None,
    logger: StructuredLogger,
) -> str:
    envelope = store.load_latest()
    if envelope is None:
        if runtime is None:
            raise MeshError(
                "E_SNAPSHOT",
                "no snapshot exists to corrupt",
                field="snapshot",
                location=str(store.directory),
            )
        envelope = store.write(runtime.snapshot_state())
    path = Path(envelope.path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["payload"]["completed_cursor"] = int(raw["payload"]["completed_cursor"]) + 1
    path.write_text(json.dumps(raw, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    logger.warning(
        "anomaly_injection",
        code="E_SNAPSHOT",
        kind="snapshot-digest",
        snapshot=str(path),
        message="payload modified without updating digest",
    )
    return str(path)


def _requests_from_ledger(ledger: Ledger, spec: MeshSpec) -> list[RequestSpec]:
    requests: list[RequestSpec] = []
    for row in ledger.request_sequence():
        payload = json.loads(row["payload_json"])
        requests.append(
            RequestSpec(
                id=str(row["request_id"]),
                service=str(row["service"]),
                payload=payload,
                order=int(row["request_seq"]),
                source_path=str(ledger.path),
            )
        )
    parsed = parse_requests(
        [
            {"id": request.id, "service": request.service, "payload": request.payload}
            for request in requests
        ],
        spec.services.keys(),
        source_path=str(ledger.path),
    )
    return parsed


def _make_runtime(
    spec: MeshSpec,
    revision: Any,
    ledger: Ledger,
    snapshot_dir: str,
    logger: StructuredLogger,
    session_id: str,
) -> MeshRuntime:
    return MeshRuntime(
        spec,
        revision,
        ledger,
        SnapshotStore(snapshot_dir, logger),
        logger,
        session_id,
    )


def _validate_ledger_spec_identity(
    spec: MeshSpec,
    meta: dict[str, str],
    ledger_path: str,
) -> None:
    recorded_instance_version = meta.get("instance_version")
    actual_instance_version = spec.instance_version()
    if recorded_instance_version and recorded_instance_version != actual_instance_version:
        raise MeshError(
            "E_SPEC",
            "loaded instance configuration does not match the version recorded in the ledger",
            field="instance_version",
            location=ledger_path,
            details={
                "recorded": recorded_instance_version,
                "loaded": actual_instance_version,
            },
        )
    recorded_config_digest = meta.get("config_digest")
    actual_config_digest = spec.config_digest()
    if recorded_config_digest and recorded_config_digest != actual_config_digest:
        raise MeshError(
            "E_SPEC",
            "loaded spec digest does not match the digest recorded in the ledger",
            field="config_digest",
            location=ledger_path,
            details={
                "recorded": recorded_config_digest,
                "loaded": actual_config_digest,
            },
        )


def command_validate(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    spec = load_spec(args.spec)
    result: dict[str, Any] = {
        "status": "ok",
        "spec": args.spec,
        "schema_version": spec.schema_version,
        "services": [
            {
                "id": service.id,
                "quota": service.quota,
                "instances": [
                    {
                        "id": instance.id,
                        "host": instance.host,
                        "port": instance.port,
                        "weight": instance.weight,
                    }
                    for instance in service.instances
                ],
            }
            for service in spec.ordered_services()
        ],
        "route_revisions": [revision.revision for revision in spec.route_revisions],
        "instance_version": spec.instance_version(),
        "config_digest": spec.config_digest(),
    }
    if args.requests:
        requests = load_requests(args.requests, spec.services.keys())
        result["requests"] = len(requests)
    logger.info("spec_validated", spec=args.spec, services=len(spec.services))
    return result


def command_check_ports(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    spec = load_spec(args.spec)
    UpstreamManager(spec).preflight()
    ports = [
        {"service": service.id, "instance": instance.id, "host": instance.host, "port": instance.port}
        for service in spec.ordered_services()
        for instance in service.instances
    ]
    logger.info("ports_checked", ports=ports)
    return {"status": "ok", "ports": ports}


def command_run(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    spec = _load_runtime_spec(args, logger)
    revision = _spec_revision(spec, args.revision)
    requests = load_requests(args.requests, spec.services.keys())
    ledger = Ledger(args.ledger, reset=args.reset, logger=logger)
    runtime: MeshRuntime | None = None
    held_port: socket.socket | None = None
    try:
        if args.inject == "port-occupied":
            held_port = _reserve_injected_port(spec, logger)
        runtime = _make_runtime(
            spec,
            revision,
            ledger,
            args.snapshot_dir,
            logger,
            args.session_id or _default_session_id("run"),
        )
        if args.inject == "snapshot-digest":
            path = _corrupt_latest_snapshot(SnapshotStore(args.snapshot_dir, logger), runtime, logger)
            SnapshotStore(args.snapshot_dir, logger).load(path)
        snapshot_validation = runtime.validate_or_replace_snapshot()
        runtime.start()
        report = runtime.run_requests(requests)
        snapshot = runtime.write_snapshot("run_complete")
        report["snapshot"] = {
            "path": snapshot.path,
            "digest": snapshot.digest,
            "validation": snapshot_validation,
        }
        report["runtime_state"] = runtime.runtime_state()
        return report
    finally:
        if held_port is not None:
            held_port.close()
        if runtime is not None:
            runtime.stop()
        ledger.close()


def command_recover(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    ledger = Ledger(args.ledger, logger=logger)
    runtime: MeshRuntime | None = None
    try:
        spec = load_spec(args.spec)
        meta = ledger.meta_dict()
        _validate_ledger_spec_identity(spec, meta, str(ledger.path))
        requested_revision = args.revision
        if requested_revision is None and meta.get("active_route_revision"):
            requested_revision = int(meta["active_route_revision"])
        revision = _spec_revision(spec, requested_revision)
        recovery = ledger.recover()
        runtime = _make_runtime(
            spec,
            revision,
            ledger,
            args.snapshot_dir,
            logger,
            args.session_id or _default_session_id("recover"),
        )
        runtime.apply_recovery(recovery)
        snapshot_validation = runtime.validate_or_replace_snapshot(apply_state=False)
        rerun_requests = _requests_from_ledger(ledger, spec)
        rerun_requests = [
            request
            for request in rerun_requests
            if any(
                str(row["request_id"]) == request.id
                for row in recovery.requests_to_rerun
            )
        ]
        report: dict[str, Any] = {
            "status": "recovered",
            "ledger": str(ledger.path),
            "exit_report_available": False,
            "pre_crash_exit_report": None,
            "recovery": {
                "budget_used_global": recovery.budget_used_global,
                "budget_used_by_service": recovery.budget_used_by_service,
                "completed_cursor": recovery.completed_cursor,
                "unknown_attempts": recovery.unknown_attempts,
                "requests_to_rerun": [str(row["request_id"]) for row in recovery.requests_to_rerun],
                "state_source": "committed_ledger_attempts_only",
            },
            "snapshot_validation": snapshot_validation,
        }
        if rerun_requests:
            runtime.start()
            run_report = runtime.run_requests(rerun_requests)
            snapshot = runtime.write_snapshot("recovery_complete")
            report["rerun"] = run_report
            report["snapshot"] = {"path": snapshot.path, "digest": snapshot.digest}
            report["status"] = "recovered_and_rerun"
        else:
            snapshot = runtime.write_snapshot("recovery_no_rerun_needed")
            report["snapshot"] = {"path": snapshot.path, "digest": snapshot.digest}
        report["runtime_state"] = runtime.runtime_state()
        return report
    finally:
        if runtime is not None:
            runtime.stop()
        ledger.close()


def command_replay(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    expected = Ledger(args.source_ledger, logger=logger)
    spec = load_spec(args.spec)
    actual: Ledger | None = None
    runtime: MeshRuntime | None = None
    try:
        meta = expected.meta_dict()
        _validate_ledger_spec_identity(spec, meta, str(expected.path))
        requested_revision = args.revision
        if requested_revision is None and meta.get("active_route_revision"):
            requested_revision = int(meta["active_route_revision"])
        revision = _spec_revision(spec, requested_revision)
        requests = (
            load_requests(args.requests, spec.services.keys())
            if args.requests
            else _requests_from_ledger(expected, spec)
        )
        actual = Ledger(args.replay_ledger, reset=True, logger=logger)
        runtime = _make_runtime(
            spec,
            revision,
            actual,
            args.snapshot_dir,
            logger,
            args.session_id or _default_session_id("replay"),
        )
        runtime.start()
        run_report = runtime.run_requests(requests)
        snapshot = runtime.write_snapshot("replay_complete")
        comparison = compare_ledgers(expected, actual)
        return {
            "status": "consistent",
            "code": "OK",
            "replay": comparison,
            "run": run_report,
            "snapshot": {"path": snapshot.path, "digest": snapshot.digest},
        }
    finally:
        if runtime is not None:
            runtime.stop()
        if actual is not None:
            actual.close()
        expected.close()


def command_ui(args: argparse.Namespace, logger: StructuredLogger) -> dict[str, Any]:
    from .ui import launch_ui

    launch_ui(
        spec_path=args.spec,
        requests_path=args.requests,
        ledger_path=args.ledger,
        snapshot_dir=args.snapshot_dir,
        revision=args.revision,
        logger=logger,
    )
    return {"status": "ui_closed"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m meshwork.cli",
        description="Deterministic local service-mesh traffic diagnostics workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_log_option(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--log-file",
            default="logs/workbench.jsonl",
            help="structured JSONL log path (default: logs/workbench.jsonl)",
        )

    validate = subparsers.add_parser("validate", help="validate a local spec")
    add_log_option(validate)
    validate.add_argument("--spec", required=True)
    validate.add_argument("--requests")

    ports = subparsers.add_parser("check-ports", help="check declared loopback ports")
    add_log_option(ports)
    ports.add_argument("--spec", required=True)

    run = subparsers.add_parser("run", help="start loopback upstreams and run a request batch")
    add_log_option(run)
    run.add_argument("--spec", required=True)
    run.add_argument("--requests", required=True)
    run.add_argument("--ledger", default="artifacts/ledger.db")
    run.add_argument("--snapshot-dir", default="snapshots")
    run.add_argument("--revision", type=int)
    run.add_argument("--session-id")
    run.add_argument("--reset", action="store_true", help="replace an existing ledger")
    run.add_argument(
        "--inject",
        choices=("corrupt-rules", "port-occupied", "snapshot-digest"),
        help="alter an input file/environment and exercise normal error paths",
    )

    recover = subparsers.add_parser("recover", help="recover committed state after a forced stop")
    add_log_option(recover)
    recover.add_argument("--spec", required=True)
    recover.add_argument("--ledger", required=True)
    recover.add_argument("--snapshot-dir", default="snapshots")
    recover.add_argument("--revision", type=int)
    recover.add_argument("--session-id")

    replay = subparsers.add_parser("replay", help="rerun a request sequence and compare ledgers")
    add_log_option(replay)
    replay.add_argument("--spec", required=True)
    replay.add_argument("--source-ledger", required=True)
    replay.add_argument("--replay-ledger", default="artifacts/replay.db")
    replay.add_argument("--requests", help="defaults to the source ledger request sequence")
    replay.add_argument("--snapshot-dir", default="snapshots/replay")
    replay.add_argument("--revision", type=int)
    replay.add_argument("--session-id")

    ui = subparsers.add_parser("ui", help="launch the Tkinter diagnostics interface")
    add_log_option(ui)
    ui.add_argument("--spec", default="samples/sample_spec.json")
    ui.add_argument("--requests", default="samples/requests.json")
    ui.add_argument("--ledger", default="artifacts/ui.db")
    ui.add_argument("--snapshot-dir", default="snapshots")
    ui.add_argument("--revision", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = _logger(args.log_file)
    handlers = {
        "validate": command_validate,
        "check-ports": command_check_ports,
        "run": command_run,
        "recover": command_recover,
        "replay": command_replay,
        "ui": command_ui,
    }
    try:
        result = handlers[args.command](args, logger)
        _print_json({"status": "ok", **result})
        return 0
    except MeshError as exc:
        logger.error(
            "command_failed",
            code=exc.code,
            message=exc.message,
            field=exc.field,
            location=exc.location,
            request_id=exc.request_id,
            service=exc.service,
            details=exc.details,
            command=args.command,
        )
        _print_json({"status": "error", **exc.to_dict()}, stream=sys.stderr)
        return 2
    except KeyboardInterrupt:
        logger.warning("command_interrupted", command=args.command)
        return 130
    except Exception as exc:
        logger.error(
            "command_exception",
            code="E_INTERNAL",
            message=str(exc),
            command=args.command,
            traceback=traceback.format_exc(),
        )
        _print_json(
            {
                "status": "error",
                "code": "E_INTERNAL",
                "message": str(exc),
            },
            stream=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
