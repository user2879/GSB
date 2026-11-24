"""Command-line entry point for the local diagnostics workbench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .canonical import canonical_json
from .engine import run_engine
from .errors import DiagnosticError, ProbeError, UiError
from .ledger import (
    LedgerReader,
    LedgerWriter,
    create_snapshot,
    recover_unknown_windows,
    restore_snapshot,
)
from .logs import append_structured_log
from .probe import LoopbackSourceSubprocess
from .replay import replay_and_compare
from .spec import load_bundle


EXIT_CODES = {
    "E_SPEC": 2,
    "E_ORDER": 3,
    "E_BUDGET": 4,
    "E_WATERMARK": 5,
    "E_REPLAY": 6,
    "E_LEDGER": 7,
    "E_SNAPSHOT": 8,
    "E_SCHEDULER": 9,
    "E_PROBE": 10,
    "E_UI": 11,
}


def _emit(value: Any) -> None:
    sys.stdout.write(canonical_json(value) + "\n")
    sys.stdout.flush()


def _log_path(value: str | None, ledger: str | None) -> str | None:
    if value:
        return value
    if ledger:
        return str(Path(ledger).with_suffix(".events.jsonl"))
    return None


def _load(args: argparse.Namespace):
    return load_bundle(args.sources, args.rules, args.budgets)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    log_path = _log_path(args.log_json, args.ledger)
    bundle = _load(args)
    writer = LedgerWriter(
        args.ledger,
        crash_after_commits=args.crash_after,
        reset_existing=args.reset,
    )
    try:
        append_structured_log(
            log_path,
            code="RUN_START",
            severity="INFO",
            message="deterministic event-time run started",
            details={
                "session_id": bundle.session_id,
                "strategy": bundle.strategy.kind,
                "spec_digest": bundle.spec_digest,
            },
        )
        result = run_engine(
            bundle,
            writer=writer,
            injection=args.inject,
        )
        output: dict[str, Any] = {
            "ok": True,
            "ledger": str(Path(args.ledger).resolve()),
            "summary": result.summary(),
        }
        append_structured_log(
            log_path,
            code="RUN_COMPLETE",
            severity="INFO",
            message="all planned windows committed",
            details=result.summary(),
        )
        if args.probe:
            output["probe"] = _run_probe(bundle)
        writer.close()
        if args.snapshot_out:
            snapshot = create_snapshot(args.ledger, args.snapshot_out)
            output["snapshot"] = snapshot
            append_structured_log(
                log_path,
                code="SNAPSHOT_CREATED",
                severity="INFO",
                message="whole snapshot created",
                details=snapshot,
            )
        return output
    except DiagnosticError as exc:
        if writer.run_initialized:
            try:
                writer.fail_run(exc)
            except Exception:
                pass
        append_structured_log(
            log_path,
            code=exc.code,
            severity="ERROR",
            message=exc.message,
            details=exc.details,
        )
        raise
    finally:
        try:
            writer.close()
        except Exception:
            pass


def _run_probe(bundle) -> dict[str, Any]:
    settings = bundle.probe
    with LoopbackSourceSubprocess(server_delay_ms=settings.server_delay_ms) as source:
        result = source.probe(
            timeout_ms=settings.timeout_ms,
            attempts=settings.attempts,
            backoff_ms=settings.backoff_ms,
        )
    return result.to_dict()


def _probe(args: argparse.Namespace) -> dict[str, Any]:
    log_path = _log_path(args.log_json, None)
    bundle = _load(args)
    settings = bundle.probe
    timeout_ms = args.timeout_ms if args.timeout_ms is not None else settings.timeout_ms
    attempts = args.attempts if args.attempts is not None else settings.attempts
    server_delay_ms = (
        args.server_delay_ms
        if args.server_delay_ms is not None
        else settings.server_delay_ms
    )
    with LoopbackSourceSubprocess(server_delay_ms=server_delay_ms) as source:
        result = source.probe(
            timeout_ms=timeout_ms,
            attempts=attempts,
            backoff_ms=settings.backoff_ms,
        )
    payload = {
        "ok": result.ok,
        "loopback_only": True,
        "timeout_ms": timeout_ms,
        "server_delay_ms": server_delay_ms,
        "result": result.to_dict(),
    }
    append_structured_log(
        log_path,
        code="PROBE_COMPLETE" if result.ok else "E_PROBE",
        severity="INFO" if result.ok else "ERROR",
        message="loopback source sample completed",
        details=payload,
    )
    return payload


def _recover(args: argparse.Namespace) -> dict[str, Any]:
    log_path = _log_path(args.log_json, args.ledger)
    unknown = recover_unknown_windows(args.ledger)
    report = {
        "ok": True,
        "ledger": str(Path(args.ledger).resolve()),
        "unknown_windows": unknown,
        "forced_exit_report_available": False,
        "required_action": "rerun" if unknown else "none",
    }
    append_structured_log(
        log_path,
        code="RECOVERY_REQUIRED" if unknown else "RECOVERY_CLEAN",
        severity="WARNING" if unknown else "INFO",
        message="crash recovery classified committed and unknown windows",
        details=report,
    )
    return report


def _status(args: argparse.Namespace) -> dict[str, Any]:
    log_path = _log_path(args.log_json, args.ledger)
    unknown = recover_unknown_windows(args.ledger)
    report = LedgerReader(args.ledger).status_report()
    report["ok"] = True
    report["ledger"] = str(Path(args.ledger).resolve())
    report["unknown_recovery"] = unknown
    append_structured_log(
        log_path,
        code="STATUS",
        severity="INFO",
        message="ledger status inspected",
        details={
            "run_status": report["meta"].get("run_status"),
            "finalized": len(report["finalized"]),
            "unknown": len(unknown),
        },
    )
    return report


def _snapshot(args: argparse.Namespace) -> dict[str, Any]:
    log_path = _log_path(args.log_json, args.ledger)
    record = create_snapshot(args.ledger, args.out, snapshot_id=args.snapshot_id)
    append_structured_log(
        log_path,
        code="SNAPSHOT_CREATED",
        severity="INFO",
        message="whole snapshot created",
        details=record,
    )
    return {"ok": True, **record}


def _restore(args: argparse.Namespace) -> dict[str, Any]:
    log_path = _log_path(args.log_json, args.ledger)
    report = restore_snapshot(args.ledger, args.snapshot)
    append_structured_log(
        log_path,
        code="SNAPSHOT_RESTORED",
        severity="INFO",
        message="snapshot replaced window conclusions and watermark state",
        details=report,
    )
    return {"ok": True, **report}


def _replay(args: argparse.Namespace) -> dict[str, Any]:
    log_path = _log_path(args.log_json, args.ledger)
    bundle = _load(args)
    report = replay_and_compare(bundle, args.ledger)
    append_structured_log(
        log_path,
        code="REPLAY_MATCH",
        severity="INFO",
        message="deterministic replay matched ledger",
        details=report,
    )
    return {"ok": True, **report}


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    bundle = _load(args)
    return {
        "ok": True,
        "session_id": bundle.session_id,
        "scale": bundle.scale,
        "strategy": bundle.strategy.to_dict(),
        "rules_version": bundle.rules_version,
        "sources": [source.source_id for source in bundle.sources],
        "rules": [rule.rule_id for rule in bundle.rules],
        "topological_rules": [rule.rule_id for rule in bundle.topological_rules],
        "spec_digest": bundle.spec_digest,
    }


def _ui(args: argparse.Namespace) -> dict[str, Any]:
    from .ui import launch_dashboard

    try:
        launch_dashboard(args.ledger)
    except UiError:
        raise
    except Exception as exc:  # noqa: BLE001 - Tk availability is environment dependent
        raise UiError(
            "Tkinter diagnostics interface could not be opened",
            details={"ledger": args.ledger, "cause": f"{type(exc).__name__}: {exc}"},
        ) from exc
    return {"ok": True, "ledger": args.ledger, "ui": "closed"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lateness-ledger",
        description="Event-time window lateness displacement and watermark ledger diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_spec_args(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--sources", nargs="+", required=True)
        command_parser.add_argument("--rules", required=True)
        command_parser.add_argument("--budgets", required=True)
        command_parser.add_argument("--log-json")

    run_parser = subparsers.add_parser("run", help="run a deterministic session")
    add_spec_args(run_parser)
    run_parser.add_argument("--ledger", required=True)
    run_parser.add_argument(
        "--inject", choices=("order", "budget", "watermark"), default=None
    )
    run_parser.add_argument("--crash-after", type=int, default=None)
    run_parser.add_argument("--snapshot-out")
    run_parser.add_argument("--reset", action="store_true")
    run_parser.add_argument("--probe", action="store_true")
    run_parser.set_defaults(handler=_run)

    probe_parser = subparsers.add_parser(
        "probe", help="sample a loopback source subprocess with real socket timeouts"
    )
    add_spec_args(probe_parser)
    probe_parser.add_argument("--timeout-ms", type=int)
    probe_parser.add_argument("--attempts", type=int)
    probe_parser.add_argument("--server-delay-ms", type=int)
    probe_parser.set_defaults(handler=_probe)

    recover_parser = subparsers.add_parser(
        "recover", help="mark planned-but-uncommitted windows unknown"
    )
    recover_parser.add_argument("--ledger", required=True)
    recover_parser.add_argument("--log-json")
    recover_parser.set_defaults(handler=_recover)

    status_parser = subparsers.add_parser("status", help="show stable ledger status")
    status_parser.add_argument("--ledger", required=True)
    status_parser.add_argument("--log-json")
    status_parser.set_defaults(handler=_status)

    snapshot_parser = subparsers.add_parser("snapshot", help="create a whole snapshot")
    snapshot_parser.add_argument("--ledger", required=True)
    snapshot_parser.add_argument("--out", required=True)
    snapshot_parser.add_argument("--snapshot-id")
    snapshot_parser.add_argument("--log-json")
    snapshot_parser.set_defaults(handler=_snapshot)

    restore_parser = subparsers.add_parser(
        "restore", help="replace conclusions and watermark state from a snapshot"
    )
    restore_parser.add_argument("--ledger", required=True)
    restore_parser.add_argument("--snapshot", required=True)
    restore_parser.add_argument("--log-json")
    restore_parser.set_defaults(handler=_restore)

    replay_parser = subparsers.add_parser("replay", help="replay and compare every ledger item")
    add_spec_args(replay_parser)
    replay_parser.add_argument("--ledger", required=True)
    replay_parser.set_defaults(handler=_replay)

    validate_parser = subparsers.add_parser("validate", help="validate the local JSON specification")
    add_spec_args(validate_parser)
    validate_parser.set_defaults(handler=_validate)

    ui_parser = subparsers.add_parser("ui", help="open the Tkinter diagnostics workbench")
    ui_parser.add_argument("--ledger", required=True)
    ui_parser.set_defaults(handler=_ui)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = args.handler(args)
        _emit(output)
        return 0
    except DiagnosticError as exc:
        _emit({"ok": False, "error": exc.to_dict()})
        return EXIT_CODES.get(exc.code, 1)
    except KeyboardInterrupt:
        _emit(
            {
                "ok": False,
                "error": {
                    "code": "E_INTERRUPTED",
                    "message": "interrupted; no exit report was produced",
                    "details": {},
                },
            }
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
