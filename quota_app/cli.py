from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .codec import parse_document
from .config_workspace import ConfigWorkspace, ConfigWorkspaceResult, open_workspace
from .derive import resolve_effective
from .errors import AppError, ConfigFormatError, E_UPGRADE
from .events import load_events
from .injections import INJECTION_KINDS, run_injection
from .ledger import Ledger
from .logging_utils import StructuredLogger
from .models import ConfigDocument, Derivation, LIMIT_DIMENSIONS, VolumeConfig
from .preview import PreviewService
from .utils import is_local_path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "data" / "configs"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "data" / "events" / "access-events.ndjson"
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "data" / "ledger.sqlite3"
DEFAULT_LOG_PATH = PROJECT_ROOT / "data" / "logs" / "quota-app.jsonl"


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _local_or_error(path: str, label: str) -> None:
    if not is_local_path(path):
        raise ConfigFormatError(
            f"{label} must be a local filesystem path",
            field_path=label,
            details={"value": path},
        )


def _open_workspace_for_command(
    args: argparse.Namespace,
    ledger: Ledger | None,
) -> ConfigWorkspaceResult:
    callback = ledger.record_upgrade_checkpoint if ledger is not None else None
    result = open_workspace(args.config_dir, checkpoint_callback=callback)
    if result.format_errors:
        raise result.format_errors[0]
    return result


def _log_workspace_errors(result: ConfigWorkspaceResult, logger: StructuredLogger, operation: str) -> None:
    for error in result.errors:
        logger.error(error, operation=operation)


def _summary(result: ConfigWorkspaceResult) -> dict[str, Any]:
    active = result.active
    summary: dict[str, Any] = {
        "directory": result.directory,
        "documents": [
            {
                "name": document.name,
                "source_path": document.source_path,
                "revision": document.revision,
                "format_version": document.schema_version,
                "upgraded_from": document.original_version,
            }
            for document in result.documents
        ],
        "active_revision": active.revision if active else None,
        "upgrade": result.upgrade.as_dict(),
    }
    if active is not None:
        effective = resolve_effective(active)
        summary["volumes"] = [
            effective[volume_id].to_json() for volume_id in sorted(effective)
        ]
        summary["format_version"] = active.schema_version
        summary["original_format_version"] = active.original_version
        summary["digest"] = active.fingerprint
    return summary


def command_validate(args: argparse.Namespace, logger: StructuredLogger) -> tuple[int, dict[str, Any]]:
    result = _open_workspace_for_command(args, None)
    _log_workspace_errors(result, logger, "validate")
    active = result.require_active()
    resolve_effective(active)
    payload = {"code": E_UPGRADE if result.upgrade.errors else "OK", **_summary(result)}
    return (1 if result.upgrade.errors else 0), payload


def command_upgrade(args: argparse.Namespace, logger: StructuredLogger) -> tuple[int, dict[str, Any]]:
    with Ledger(args.ledger) as ledger:
        result = _open_workspace_for_command(args, ledger)
        _log_workspace_errors(result, logger, "upgrade")
        payload = {
            "code": E_UPGRADE if result.upgrade.errors else "OK",
            "upgrade": result.upgrade.as_dict(),
            "active_revision": result.active.revision if result.active else None,
        }
    return (1 if result.upgrade.errors else 0), payload


def command_run(args: argparse.Namespace, logger: StructuredLogger) -> tuple[int, dict[str, Any]]:
    if args.inject:
        injection = run_injection(
            args.inject,
            config_dir=args.config_dir,
            events_path=args.events,
        )
        if injection.error:
            logger.emit(
                injection.observed_code or "E_INTERNAL",
                injection.error.get("message", "injected failure"),
                operation=f"inject.{args.inject}",
                volume_id=injection.error.get("volume_id"),
                field_path=injection.error.get("field_path"),
                details=injection.error.get("details"),
            )
        payload = {
            "code": injection.observed_code or "E_INTERNAL",
            **injection.as_dict(),
        }
        return (0 if injection.passed else 1), payload

    with Ledger(args.ledger) as ledger:
        workspace = _open_workspace_for_command(args, ledger)
        _log_workspace_errors(workspace, logger, "run.workspace")
        active = workspace.require_active()
        replaced = ledger.register_config(active, source=active.source_path or "workspace")
        events = load_events(args.events)
        recovery = ledger.recover_incomplete()
        if recovery.unknown_batches:
            logger.emit(
                "RECOVERY_PARTIAL",
                "unfinished batches were marked unknown; rerun is required",
                operation="run.recover",
                details=recovery.as_dict(),
            )
        replay_before = ledger.verify_replay()
        if not replay_before.ok:
            assert replay_before.errors
            logger.error(replay_before.errors[0], operation="run.replay.before")
            raise replay_before.errors[0]
        ingest = ledger.ingest_events(active, events, batch_size=args.batch_size)
        replay_after = ledger.verify_replay()
        if not replay_after.ok:
            assert replay_after.errors
            logger.error(replay_after.errors[0], operation="run.replay.after")
            raise replay_after.errors[0]
        view = ledger.current_view()
        code = E_UPGRADE if workspace.upgrade.errors else "OK"
        payload = {
            "code": code,
            "active_revision": active.revision,
            "config_replaced_wholesale": replaced,
            "recovery": recovery.as_dict(),
            "ingest": ingest.as_dict(),
            "replay": replay_after.as_dict(),
            "workspace": _summary(workspace),
            "view": view,
        }
        return (1 if workspace.upgrade.errors else 0), payload


def command_recover(args: argparse.Namespace, logger: StructuredLogger) -> tuple[int, dict[str, Any]]:
    with Ledger(args.ledger) as ledger:
        recovery = ledger.recover_incomplete()
        replay = ledger.verify_replay()
        if not replay.ok:
            assert replay.errors
            logger.error(replay.errors[0], operation="recover.replay")
            raise replay.errors[0]
        payload = {
            "code": "OK",
            "recovery": recovery.as_dict(),
            "replay": replay.as_dict(),
            "report": ledger.current_view(),
            "note": (
                "exit-time reporting is unavailable while unknown batches exist"
                if recovery.rerun_required
                else "all committed batches are reproducible"
            ),
        }
        return 0, payload


def _candidate_document(
    args: argparse.Namespace,
    workspace: ConfigWorkspaceResult,
    *,
    revision: int | None = None,
) -> ConfigDocument:
    active = workspace.require_active()
    if not args.volume:
        return active
    volume = active.volume_map.get(args.volume)
    if volume is None:
        raise ConfigFormatError(
            f"volume {args.volume!r} does not exist",
            volume_id=args.volume,
            field_path="$.volumes",
        )
    derivation = volume.derivation
    if args.upstream is not None or args.factor is not None or args.offset is not None:
        upstream = args.upstream if args.upstream is not None else (derivation.upstream_id if derivation else None)
        factor = args.factor if args.factor is not None else (derivation.factor if derivation else 1)
        offset = args.offset if args.offset is not None else (derivation.offset if derivation else 0)
        if upstream is None:
            raise ConfigFormatError(
                "--upstream is required when creating a derivation",
                volume_id=args.volume,
                field_path="$.volumes[].derivation.upstream_id",
            )
        derivation = Derivation(upstream_id=upstream, factor=factor, offset=offset)
    updated = VolumeConfig(
        volume_id=volume.volume_id,
        capacity_bytes=args.capacity_bytes if args.capacity_bytes is not None else volume.capacity_bytes,
        ops_per_sec=args.ops_per_sec if args.ops_per_sec is not None else volume.ops_per_sec,
        burst_capacity=args.burst_capacity if args.burst_capacity is not None else volume.burst_capacity,
        limit_dimension=args.limit_dimension or volume.limit_dimension,
        derivation=derivation,
    )
    candidate = active.with_volume(
        updated,
        revision=active.revision + 1 if revision is None else revision,
    )
    parsed = parse_document(candidate.to_json(), source_path=active.source_path)
    resolve_effective(parsed)
    return parsed


def command_edit(args: argparse.Namespace, logger: StructuredLogger) -> tuple[int, dict[str, Any]]:
    with Ledger(args.ledger) as ledger:
        workspace = ConfigWorkspace(args.config_dir, checkpoint_callback=ledger.record_upgrade_checkpoint)
        result = workspace.open()
        _log_workspace_errors(result, logger, "edit.workspace")
        if result.format_errors:
            raise result.format_errors[0]
        active = result.require_active()
        ledger.register_config(active)
        ledger_revision = ledger.active_revision() or 0
        revision = workspace.next_revision(minimum=ledger_revision)
        candidate = _candidate_document(args, result, revision=revision)
        workspace.save_active(candidate, minimum_revision=ledger_revision)
        replaced = ledger.register_config(candidate)

        ingest_payload: dict[str, Any] | None = None
        if args.ingest:
            events = load_events(args.events)
            ledger.recover_incomplete()
            ingest = ledger.ingest_events(candidate, events, batch_size=args.batch_size)
            replay = ledger.verify_replay()
            if not replay.ok:
                assert replay.errors
                logger.error(replay.errors[0], operation="edit.replay")
                raise replay.errors[0]
            ingest_payload = {"ingest": ingest.as_dict(), "replay": replay.as_dict()}

        replacements = ledger.list_replacements()
        payload: dict[str, Any] = {
            "code": E_UPGRADE if result.upgrade.errors else "OK",
            "active_revision": candidate.revision,
            "source_path": candidate.source_path,
            "config_replaced_wholesale": replaced,
            "replacement": replacements[-1] if replacements else None,
        }
        if ingest_payload is not None:
            payload.update(ingest_payload)
        return (1 if result.upgrade.errors else 0), payload


def command_preview(args: argparse.Namespace, logger: StructuredLogger) -> tuple[int, dict[str, Any]]:
    workspace = _open_workspace_for_command(args, None)
    _log_workspace_errors(workspace, logger, "preview.workspace")
    candidate = _candidate_document(args, workspace)
    with Ledger(args.ledger) as ledger:
        active = workspace.require_active()
        ledger.register_config(active)
        events = ledger.load_event_inputs()
        if not events:
            events = load_events(args.events)
        service = PreviewService(events, logger=logger, budget_ms=args.preview_budget_ms)
        preview, authoritative, comparison = service.run(candidate)
        code = "OK"
        if comparison.mismatches:
            code = "E_PREVIEW"
        elif preview.latency_exceeded:
            code = "E_LATENCY"
        payload = {
            "code": code,
            "candidate_revision": candidate.revision,
            "preview": preview.as_dict(),
            "authoritative": authoritative.as_dict(),
            "comparison": comparison.as_dict(),
        }
        return (0 if code == "OK" else 1), payload


def command_inject(args: argparse.Namespace, logger: StructuredLogger) -> tuple[int, dict[str, Any]]:
    injection = run_injection(
        args.kind,
        config_dir=args.config_dir,
        events_path=args.events,
    )
    if injection.error:
        logger.emit(
            injection.observed_code or "E_INTERNAL",
            injection.error.get("message", "injected failure"),
            operation=f"inject.{args.kind}",
            volume_id=injection.error.get("volume_id"),
            field_path=injection.error.get("field_path"),
            details=injection.error.get("details"),
        )
    payload = {"code": injection.observed_code or "E_INTERNAL", **injection.as_dict()}
    return (0 if injection.passed else 1), payload


def command_view(args: argparse.Namespace, logger: StructuredLogger) -> tuple[int, dict[str, Any]]:
    with Ledger(args.ledger) as ledger:
        payload = {"code": "OK", "view": ledger.current_view()}
    return 0, payload


def command_gui(args: argparse.Namespace, logger: StructuredLogger) -> tuple[int, dict[str, Any]]:
    from .gui import launch_gui

    launch_gui(
        config_dir=args.config_dir,
        events_path=args.events,
        ledger_path=args.ledger,
        log_path=args.log,
        preview_budget_ms=args.preview_budget_ms,
    )
    return 0, {"code": "OK", "message": "GUI closed"}


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    common.add_argument("--events", default=str(DEFAULT_EVENTS_PATH))
    common.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    common.add_argument("--log", default=str(DEFAULT_LOG_PATH))
    common.add_argument("--json", action="store_true", help="accepted for explicit JSON scripting")

    parser = argparse.ArgumentParser(
        prog="quota-app",
        description="Local storage quota and rate-limit configuration editor",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", parents=[common], help="validate and normalize configs")
    validate.set_defaults(handler=command_validate)

    upgrade = subparsers.add_parser("upgrade", parents=[common], help="upgrade legacy config entries")
    upgrade.set_defaults(handler=command_upgrade)

    run = subparsers.add_parser("run", parents=[common], help="ingest events into the SQLite ledger")
    run.add_argument("--batch-size", type=int, default=3)
    run.add_argument("--preview-budget-ms", type=int, default=50)
    run.add_argument("--inject", choices=INJECTION_KINDS)
    run.add_argument(
        "--inject-corrupt-config",
        action="store_const",
        dest="inject",
        const="corrupt-config",
    )
    run.add_argument(
        "--inject-missing-checkpoint",
        action="store_const",
        dest="inject",
        const="missing-checkpoint",
    )
    run.add_argument(
        "--inject-timestamp-regression",
        action="store_const",
        dest="inject",
        const="timestamp-regression",
    )
    run.set_defaults(handler=command_run)

    recover = subparsers.add_parser("recover", parents=[common], help="recover and replay committed batches")
    recover.set_defaults(handler=command_recover)

    preview = subparsers.add_parser("preview", parents=[common], help="fast preview and authoritative replay")
    preview.add_argument("--volume")
    preview.add_argument("--capacity-bytes", type=int)
    preview.add_argument("--ops-per-sec", type=int)
    preview.add_argument("--burst-capacity", type=int)
    preview.add_argument("--limit-dimension", choices=LIMIT_DIMENSIONS)
    preview.add_argument("--upstream")
    preview.add_argument("--factor", type=int)
    preview.add_argument("--offset", type=int)
    preview.add_argument("--preview-budget-ms", type=int, default=50)
    preview.set_defaults(handler=command_preview)

    edit = subparsers.add_parser("edit", parents=[common], help="persist a monotonically versioned edit")
    edit.add_argument("--volume", required=True)
    edit.add_argument("--capacity-bytes", type=int)
    edit.add_argument("--ops-per-sec", type=int)
    edit.add_argument("--burst-capacity", type=int)
    edit.add_argument("--limit-dimension", choices=LIMIT_DIMENSIONS)
    edit.add_argument("--upstream")
    edit.add_argument("--factor", type=int)
    edit.add_argument("--offset", type=int)
    edit.add_argument("--ingest", action="store_true")
    edit.add_argument("--batch-size", type=int, default=3)
    edit.set_defaults(handler=command_edit)

    view = subparsers.add_parser("view", parents=[common], help="print the current ledger view")
    view.set_defaults(handler=command_view)

    inject = subparsers.add_parser("inject", parents=[common], help="run a local anomaly injection")
    inject.add_argument("kind", choices=INJECTION_KINDS)
    inject.set_defaults(handler=command_inject)

    gui = subparsers.add_parser("gui", parents=[common], help="launch the Tkinter editor")
    gui.add_argument("--preview-budget-ms", type=int, default=50)
    gui.set_defaults(handler=command_gui)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for path, label in (
        (args.config_dir, "config-dir"),
        (args.events, "events"),
        (args.ledger, "ledger"),
        (args.log, "log"),
    ):
        try:
            _local_or_error(path, label)
        except AppError as exc:
            print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True, indent=2))
            return 1

    logger = StructuredLogger(args.log)
    operation = args.command
    try:
        code, payload = args.handler(args, logger)
        _emit(payload)
        return code
    except AppError as exc:
        logger.error(exc, operation=operation)
        _emit({"code": exc.code, **exc.as_dict()})
        return 1
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        logger.emit(
            "E_INTERNAL",
            str(exc),
            operation=operation,
            details={"exception_type": type(exc).__name__},
        )
        _emit(
            {
                "code": "E_INTERNAL",
                "message": str(exc),
                "exception_type": type(exc).__name__,
            }
        )
        return 2
