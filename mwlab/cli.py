"""Command-line interface for the interoperability experiments."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .artifacts import (
    assert_expected_summary,
    build_artifact,
    summary_from_artifact,
    write_analysis_outputs,
)
from .canonical import append_jsonl, canonical_json, read_json, utc_now, write_json
from .config import Declarations, load_declarations
from .errors import EXIT_CODES, LabError, state_error
from .independent import verify as independent_verify
from .ledger import Ledger
from .records import (
    duplicate_idempotency_details,
    load_normalized_records,
    model_digest,
)
from .selftest import run_selfcheck
from .semantics import analyze
from .snapshot import list_snapshots, restore_snapshot, save_snapshot


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-file", default=None, help="structured JSONL error log path")
    parser.add_argument("--pretty", action="store_true", help="pretty-print successful JSON output")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mwlab",
        description="Deterministic message-delivery semantics interoperability lab",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    selfcheck = subparsers.add_parser("selfcheck", help="read-only Python/sqlite/tkinter/NumPy check")
    selfcheck.add_argument("--config", default=None)
    selfcheck.add_argument("--require-tkinter", action="store_true")
    _add_output_options(selfcheck)

    init = subparsers.add_parser("init", help="create and validate ledger/snapshot/output paths")
    init.add_argument("--config", required=True)
    _add_output_options(init)

    run = subparsers.add_parser("run", help="normalize records, derive semantics, and commit topic batches")
    run.add_argument("--config", required=True)
    run.add_argument("--records", required=True)
    run.add_argument("--ledger", default=None)
    run.add_argument("--out", default=None)
    run.add_argument("--expected", default=None, help="expected summary JSON")
    run.add_argument("--inject-format", action="store_true", help="inject a negative retry_count and raise E_FORMAT")
    run.add_argument("--inject-dupkey", action="store_true", help="escalate a cross-group idempotency duplicate to E_DUPKEY")
    run.add_argument(
        "--crash-after-topic",
        type=int,
        default=None,
        metavar="N",
        help="commit N topic batches and force-exit with status 91",
    )
    _add_output_options(run)

    report = subparsers.add_parser("report", help="render fixed-point statistics and conflict attribution")
    report.add_argument("--artifact", required=True)
    report.add_argument("--out", default=None)
    _add_output_options(report)

    verify = subparsers.add_parser("verify", help="independent normalization/semantics/statistics replay")
    verify.add_argument("--config", required=True)
    verify.add_argument("--records", required=True)
    verify.add_argument("--model-digest", required=True)
    verify.add_argument("--conclusions", required=True)
    verify.add_argument("--group-conclusions", required=True)
    verify.add_argument("--conflicts", required=True)
    verify.add_argument("--stats", required=True)
    verify.add_argument("--inject-divergence", action="store_true")
    _add_output_options(verify)

    recover = subparsers.add_parser("recover", help="recover committed topic batches and replay them")
    recover.add_argument("--ledger", default=None)
    recover.add_argument("--config", default=None)
    recover.add_argument("--records", default=None)
    _add_output_options(recover)

    snapshot = subparsers.add_parser("snapshot", help="save, restore, or list versioned snapshots")
    snapshot_sub = snapshot.add_subparsers(dest="snapshot_action", required=True)
    for action in ("save", "restore", "list"):
        child = snapshot_sub.add_parser(action)
        child.add_argument("--config", default=None)
        child.add_argument("--ledger", default=None)
        child.add_argument("--dir", default=None, help="snapshot directory")
        child.add_argument("--backup-dir", default=None)
        if action != "list":
            child.add_argument("--name", required=True)
        if action == "restore":
            child.add_argument("--inject-digest", action="store_true")
        _add_output_options(child)
    return parser


def _emit(value: Any, *, pretty: bool = False) -> None:
    if pretty:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(canonical_json(value))


def _log_path(args: argparse.Namespace, declarations: Declarations | None = None) -> Path:
    explicit = getattr(args, "log_file", None)
    if explicit:
        return Path(explicit)
    if declarations is not None:
        return declarations.paths["logs"]
    return Path("logs/structured.jsonl")


def _log_event(path: Path, event: str, payload: dict[str, Any]) -> None:
    record = {
        "timestamp_utc": utc_now(),
        "level": "ERROR" if event.endswith("error") else "INFO",
        "event": event,
        "pid": os.getpid(),
    }
    record.update(payload)
    append_jsonl(path, record)


def _ledger_path(args: argparse.Namespace, declarations: Declarations) -> Path:
    return Path(args.ledger) if getattr(args, "ledger", None) else declarations.paths["ledger"]


def _out_path(args: argparse.Namespace, declarations: Declarations) -> Path:
    return Path(args.out) if getattr(args, "out", None) else declarations.paths["out"]


def _snapshot_paths(args: argparse.Namespace, declarations: Declarations | None) -> tuple[Path, Path]:
    if getattr(args, "dir", None):
        snapshot_dir = Path(args.dir)
    elif declarations is not None:
        snapshot_dir = declarations.paths["snapshots"]
    else:
        snapshot_dir = Path("snapshots")
    if getattr(args, "backup_dir", None):
        backup_dir = Path(args.backup_dir)
    elif declarations is not None:
        backup_dir = declarations.paths["backup"]
    else:
        backup_dir = Path("backup")
    return snapshot_dir, backup_dir


def _check_distinct_paths(declarations: Declarations, ledger_path: Path, out_path: Path) -> None:
    snapshot_dir = declarations.paths["snapshots"].resolve()
    backup_dir = declarations.paths["backup"].resolve()
    ledger_resolved = ledger_path.resolve()
    out_resolved = out_path.resolve()
    conflicts = []
    if (
        snapshot_dir == backup_dir
        or backup_dir in snapshot_dir.parents
        or snapshot_dir in backup_dir.parents
    ):
        conflicts.append("snapshots path must not equal or live under the read-only backup directory")
    if ledger_resolved == backup_dir or backup_dir in ledger_resolved.parents:
        conflicts.append("ledger path must not live under the read-only backup directory")
    if out_resolved == backup_dir or backup_dir in out_resolved.parents:
        conflicts.append("output path must not live under the read-only backup directory")
    if conflicts:
        raise state_error("path layout conflicts with backup directory", conflicts=conflicts)


def _write_probe(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".write-probe-{os.getpid()}"
    try:
        probe.write_text("ok", encoding="utf-8")
        if probe.read_text(encoding="utf-8") != "ok":
            raise OSError("probe read-back mismatch")
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass


def _handle_selfcheck(args: argparse.Namespace) -> dict[str, Any]:
    declarations = load_declarations(args.config) if args.config else None
    result = run_selfcheck(declarations, require_tkinter=args.require_tkinter)
    return result


def _handle_init(args: argparse.Namespace) -> dict[str, Any]:
    declarations = load_declarations(args.config)
    ledger_path = _ledger_path(args, declarations)
    out_path = _out_path(args, declarations)
    _check_distinct_paths(declarations, ledger_path, out_path)
    writable_targets = {
        "ledger_parent": ledger_path.parent,
        "snapshots": declarations.paths["snapshots"],
        "backup": declarations.paths["backup"],
        "out": out_path,
        "logs_parent": declarations.paths["logs"].parent,
    }
    for target in writable_targets.values():
        _write_probe(target)
    return {
        "status": "initialized",
        "paths": {name: str(path) for name, path in writable_targets.items()},
        "backup_is_separate_readonly_area": True,
        "parameter_fingerprint": declarations.parameter_fingerprint(),
    }


def _inject_duplicate_key(records: list[Any], declarations: Declarations) -> dict[str, Any]:
    details = duplicate_idempotency_details(records)
    if details is not None:
        details["injected"] = True
        return details
    v3_groups = sorted(
        group.name for group in declarations.groups.values() if group.format_version == "v3"
    )
    if len(v3_groups) < 2:
        raise LabError(
            "E_DUPKEY",
            "cannot inject a cross-group idempotency duplicate without two v3 consumer groups",
            {"v3_groups": v3_groups},
        )
    source = next((record for record in records if record.source_format == "v3"), None)
    if source is None:
        raise LabError("E_DUPKEY", "cannot inject a duplicate without a v3 source record", {})
    other_group = next(group for group in v3_groups if group != source.consumer_group)
    duplicate = replace(
        source,
        source_group=other_group,
        consumer_group=other_group,
        line_no=source.line_no,
    )
    details = duplicate_idempotency_details([*records, duplicate])
    if details is None:
        raise LabError("E_DUPKEY", "duplicate-key injection failed", {})
    details["injected"] = True
    details["synthetic_duplicate"] = True
    return details


def _handle_run(args: argparse.Namespace) -> dict[str, Any]:
    declarations = load_declarations(args.config)
    records = load_normalized_records(args.records, declarations, inject_format=args.inject_format)
    if args.inject_dupkey:
        details = _inject_duplicate_key(records, declarations)
        groups = ", ".join(details["consumer_groups"])
        raise LabError(
            "E_DUPKEY",
            f"idempotency key appears in multiple consumer groups: {groups}",
            details,
        )
    analysis = analyze(records, declarations)
    ledger_path = _ledger_path(args, declarations)
    out_path = _out_path(args, declarations)
    ledger = Ledger(ledger_path)
    execution_info: dict[str, Any] | None = None
    try:
        execution_info = ledger.start_execution(
            declarations,
            model_digest(records),
            analysis["conclusions"],
        )
        artifact = build_artifact(
            declarations,
            records,
            analysis,
            parameter_version=execution_info["parameter_version"],
            execution_id=execution_info["execution_id"],
            view_diff=execution_info["view_diff"],
        )
        topic_commits = []
        for topic_index, topic in enumerate(declarations.topic_names):
            if args.crash_after_topic is not None and topic_index >= args.crash_after_topic:
                log_path = _log_path(args, declarations)
                _log_event(
                    log_path,
                    "simulated_force_termination",
                    {
                        "command": "run",
                        "execution_id": execution_info["execution_id"],
                        "committed_topic_count": topic_index,
                        "committed_topics": [row["topic"] for row in topic_commits],
                        "exit_status": 91,
                    },
                )
                os._exit(91)
            topic_records = [record for record in records if record.topic == topic]
            topic_conclusions = [row for row in artifact["conclusions"] if row["topic"] == topic]
            topic_groups = [row for row in artifact["group_conclusions"] if row["topic"] == topic]
            topic_conflicts = [row for row in artifact["conflicts"] if row["topic"] == topic]
            topic_group_stats = [
                row for row in artifact["stats"]["by_topic_group"] if row["topic"] == topic
            ]
            topic_stat = next(
                (row for row in artifact["stats"]["topic_ranking"] if row["topic"] == topic),
                None,
            )
            topic_commits.append(
                ledger.commit_topic(
                    execution_info["execution_id"],
                    topic,
                    topic_index,
                    topic_records,
                    topic_conclusions,
                    topic_groups,
                    topic_conflicts,
                    topic_group_stats,
                    topic_stat,
                )
            )
        finished = ledger.finish_execution(execution_info["execution_id"])
    finally:
        ledger.close()

    artifact["ledger_content_digest"] = finished["content_digest"]
    if args.expected:
        assert_expected_summary(artifact, read_json(args.expected))
    output_paths = write_analysis_outputs(out_path, artifact, records)
    summary = summary_from_artifact(artifact)
    summary.update(
        {
            "status": "complete",
            "ledger": str(ledger_path),
            "ledger_content_digest": finished["content_digest"],
            "parameter_changed": execution_info["parameter_changed"],
            "previous_parameter_version": execution_info["previous_parameter_version"],
            "committed_topics": [row["topic"] for row in topic_commits],
            "topic_batches": topic_commits,
            "output_files": output_paths,
        }
    )
    return summary


def _handle_report(args: argparse.Namespace) -> dict[str, Any]:
    artifact = read_json(args.artifact)
    report = {
        "status": "ok",
        "parameter_version": artifact.get("parameter_version"),
        "conclusions_digest": artifact.get("conclusions_digest"),
        "conflicts_digest": artifact.get("conflicts_digest"),
        "stats": artifact.get("stats"),
        "conflicts": artifact.get("conflicts"),
        "view_diff": artifact.get("view_diff"),
    }
    if args.out:
        write_json(args.out, report, pretty=True)
        report["output_file"] = str(args.out)
    return report


def _handle_verify(args: argparse.Namespace) -> dict[str, Any]:
    return independent_verify(
        args.config,
        args.records,
        args.model_digest,
        args.conclusions,
        args.group_conclusions,
        args.conflicts,
        args.stats,
        inject_divergence=args.inject_divergence,
    )


def _handle_recover(args: argparse.Namespace) -> dict[str, Any]:
    declarations = load_declarations(args.config) if args.config else None
    if args.ledger:
        ledger_path = Path(args.ledger)
    elif declarations is not None:
        ledger_path = declarations.paths["ledger"]
    else:
        ledger_path = Path("state/ledger.sqlite3")
    if not ledger_path.exists():
        raise state_error("ledger file does not exist", path=str(ledger_path))
    ledger = Ledger(ledger_path)
    try:
        result = ledger.recover(args.records)
    finally:
        ledger.close()
    result["ledger"] = str(ledger_path)
    return result


def _handle_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    declarations = load_declarations(args.config) if args.config else None
    snapshot_dir, backup_dir = _snapshot_paths(args, declarations)
    ledger_path = _ledger_path(args, declarations) if declarations is not None else Path("state/ledger.sqlite3")
    action = args.snapshot_action
    if action == "save":
        return save_snapshot(ledger_path, snapshot_dir, backup_dir, args.name)
    if action == "restore":
        return restore_snapshot(
            ledger_path,
            snapshot_dir,
            backup_dir,
            args.name,
            inject_digest_mismatch=args.inject_digest,
        )
    return {
        "status": "ok",
        "snapshot_dir": str(snapshot_dir),
        "snapshots": list_snapshots(snapshot_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command
    if command == "snapshot":
        command = f"snapshot_{args.snapshot_action}"
    declarations: Declarations | None = None
    try:
        if command == "selfcheck":
            result = _handle_selfcheck(args)
            _emit(result, pretty=args.pretty)
            return 0 if result["status"] != "fail" else 1
        if command == "init":
            result = _handle_init(args)
        elif command == "run":
            result = _handle_run(args)
        elif command == "report":
            result = _handle_report(args)
        elif command == "verify":
            result = _handle_verify(args)
        elif command == "recover":
            result = _handle_recover(args)
        elif command == "snapshot_save":
            result = _handle_snapshot(args)
        elif command == "snapshot_restore":
            result = _handle_snapshot(args)
        elif command == "snapshot_list":
            result = _handle_snapshot(args)
        else:
            raise state_error("unknown command", command=command)
        _emit(result, pretty=args.pretty)
        return 0
    except LabError as exc:
        log_path = _log_path(args, declarations)
        payload = {
            "command": command,
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        }
        _log_event(log_path, "command_error", payload)
        _emit({"status": "error", "error": exc.to_dict()})
        return exc.exit_code
    except KeyboardInterrupt:
        error = LabError("E_INTERNAL", "interrupted by user", {})
        _log_event(_log_path(args, declarations), "command_error", {"command": command, **error.to_dict()})
        _emit({"status": "error", "error": error.to_dict()})
        return 130
    except Exception as exc:
        error = LabError(
            "E_INTERNAL",
            "unexpected internal failure",
            {"type": type(exc).__name__, "reason": str(exc)},
        )
        _log_event(_log_path(args, declarations), "command_error", {"command": command, **error.to_dict()})
        _emit({"status": "error", "error": error.to_dict()})
        return EXIT_CODES["E_INTERNAL"]


if __name__ == "__main__":
    raise SystemExit(main())
