from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .errors import (
    E_FORMAT,
    E_MIGRATE,
    E_SNAPSHOT,
    E_VERIFY,
    FormatError,
    MigrateError,
    SchedulerError,
    SnapshotError,
    VerifyError,
    WidthError,
)
from .invalidation import compute_invalidations
from .ledger import (
    begin_session,
    commit_pod_result,
    entry_from_result,
    ledger_status,
    load_entries_readonly,
    recover_ledger,
    sync_snapshot,
)
from .logging_utils import StructuredLogger
from .migration import migrate_records
from .records import NormalizationParams, load_records
from .samples import write_sample
from .scheduler import (
    compare_results_stable,
    conclusion_bytes as result_conclusion_bytes,
    schedule_full,
    schedule_incremental,
)
from .scoring import ScoringManager
from .spec import load_spec
from .util import atomic_write_text, canonical_json, read_json
from .verifier import verify_ledger


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _load_records_with_injection(
    path: str | Path,
    *,
    spec_path: str | Path,
    inject: str | None,
) -> Any:
    spec = load_spec(spec_path)
    params = NormalizationParams.from_spec(spec)
    if inject in {None, "none"}:
        return load_records(path, params=params)
    text = Path(path).read_text(encoding="utf-8")
    if inject == "record-corruption":
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            if line.strip().startswith("{"):
                lines[index] = line[:-1] + "]"
            else:
                lines[index] = "v9" + line[2:]
            break
        corrupted = "\n".join(lines) + "\n"
    elif inject == "v1-width":
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("v1"):
                lines[index] = line + "X"
                break
        else:
            raise WidthError(
                "no v1 fixed-width line is available for width injection",
                source_path=str(path),
            )
        corrupted = "\n".join(lines) + "\n"
    else:
        raise ValueError(f"unknown record injection {inject!r}")
    fd, name = tempfile.mkstemp(
        prefix=".injected-", suffix=".jsonl", dir=str(Path(path).parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(corrupted)
        return load_records(name, params=params)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _record_summary(record: Any) -> dict[str, Any]:
    return {
        "source_format_version": record.source_format_version,
        "source_line": record.source_line,
        "pod_id": record.pod_id,
        "node_name": record.node_name,
        "total_score": record.total_score,
        "veto_code": record.veto_code,
        "parameter_version": record.parameter_version,
        "scale_exponent": record.scale_exponent,
        "scale_factor": record.scale_factor,
        "plugin_scores": dict(record.plugin_scores),
        "plugin_contributions": dict(record.plugin_contributions),
        "error_bound": record.error_bound,
        "tie_break": list(record.tie_break),
        "filled_fields": list(record.filled_fields),
        "completion_basis": {
            key: dict(value) for key, value in sorted(record.completion_basis.items())
        },
        "conclusion_bytes_sha256": hashlib.sha256(record.conclusion_bytes()).hexdigest(),
    }


def _verify_migration_output(source_backup: Path, target: Path, spec_path: Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    params = NormalizationParams.from_spec(spec)
    before = load_records(source_backup, params=params)
    after = load_records(target, params=params)
    before_records = sorted(
        before.records,
        key=lambda record: (record.source_format_version, record.source_line),
    )
    after_records = list(after.records)
    if len(before_records) != len(after_records):
        raise MigrateError(
            "migration changed the number of records",
            details={"before": len(before_records), "after": len(after_records)},
        )
    for index, (old, new) in enumerate(zip(before_records, after_records), start=1):
        if old.conclusion_bytes() == new.conclusion_bytes():
            continue
        old_value = old.conclusion()
        new_value = new.conclusion()
        for field in old_value:
            if old_value[field] != new_value.get(field):
                raise MigrateError(
                    "migration byte comparison found a differing field",
                    pod_id=old.pod_id,
                    details={
                        "ordinal": index,
                        "field": field,
                        "before": old_value[field],
                        "after": new_value.get(field),
                    },
                )
        raise MigrateError(
            "migration conclusion bytes differ without a field-level explanation",
            pod_id=old.pod_id,
            details={"ordinal": index},
        )
    return {
        "records_compared": len(before_records),
        "conclusion_bytes_equal": True,
        "comparison_mode": "canonical_conclusion_bytes_by_migration_ordinal",
    }


def _entries_from_state(
    state: dict[str, Any],
    *,
    format_versions: list[int | str],
    session_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for cursor, pod_id in enumerate(sorted(state["results"]), start=1):
        result = state["results"][pod_id]
        entries[pod_id] = entry_from_result(
            result,
            format_versions=format_versions,
            status="committed",
            cursor=cursor,
            session_id=session_id,
        )
    return entries


def cmd_generate_sample(args: argparse.Namespace) -> dict[str, Any]:
    info = write_sample(args.root, port_base=args.port_base)
    return {"ok": True, **info}


def cmd_validate_spec(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    return {
        "ok": True,
        "source_path": spec.source_path,
        "parameter_version": spec.parameter_version,
        "scale_exponent": spec.scale_exponent,
        "scale_factor": spec.scale_factor,
        "weights_myriad": dict(spec.weights_myriad),
        "node_count": len(spec.nodes),
        "pod_count": len(spec.pods),
        "service_groups": [
            {"name": group.name, "port": group.port, "nodes": list(group.node_names)}
            for group in spec.service.groups
        ],
    }


def cmd_normalize_records(args: argparse.Namespace) -> dict[str, Any]:
    batch = _load_records_with_injection(
        args.records, spec_path=args.spec, inject=args.inject
    )
    return {
        "ok": True,
        "source_path": batch.source_path,
        "format_versions": list(batch.format_versions),
        "record_count": len(batch.records),
        "records": [_record_summary(record) for record in batch.records],
    }


def cmd_migrate(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    before = load_records(
        args.records, params=NormalizationParams.from_spec(spec)
    )
    result = migrate_records(
        args.records,
        spec,
        args.db,
        batch_size=args.batch_size,
        backup_path=args.backup,
        abort_after_batches=args.abort_after_batches,
    )
    result = dict(result)
    result["source_format_versions"] = list(before.format_versions)
    backup = result.get("backup")
    if backup and Path(backup).exists():
        result["migration_verification"] = _verify_migration_output(
            Path(backup), Path(args.records), Path(args.spec)
        )
    result["ok"] = True
    return result


def cmd_schedule(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    record_batch = load_records(
        args.records, params=NormalizationParams.from_spec(spec)
    )
    format_versions = list(record_batch.format_versions)
    previous: dict[str, Any] | None = None
    if args.mode == "incremental" and Path(args.state).exists():
        previous = read_json(args.state)
    if previous is None and args.mode == "incremental":
        mode_used = "full-initial"
        affected = tuple(sorted(spec.pod_by_id))
        invalidation = compute_invalidations(None, spec)
    elif args.mode == "full":
        mode_used = "full"
        affected = tuple(sorted(spec.pod_by_id))
        invalidation = None
    else:
        mode_used = "incremental"
        assert previous is not None
        invalidation = compute_invalidations(previous, spec)
        affected = invalidation.affected_pods

    session_id = begin_session(
        args.ledger,
        sorted(spec.pod_by_id),
        format_versions=format_versions,
        parameter_version=spec.parameter_version,
        scale_factor=spec.scale_factor,
    )
    with ScoringManager(spec) as manager:
        if mode_used == "incremental":
            assert previous is not None
            state = schedule_incremental(spec, manager, previous, affected)
        else:
            state = schedule_full(spec, manager)

    if previous is not None:
        for pod_id in sorted(set(previous.get("results", {})) - set(affected)):
            if pod_id not in state["results"]:
                continue
            old_result = previous["results"][pod_id]
            new_result = state["results"][pod_id]
            if result_conclusion_bytes(old_result) != result_conclusion_bytes(new_result):
                raise VerifyError(
                    "incremental run changed an unaffected pod conclusion",
                    pod_id=pod_id,
                    node_name=str(new_result.get("node_name", "-")),
                    details={"field": "stable_conclusion"},
                )

    for cursor, pod_id in enumerate(sorted(state["results"]), start=1):
        commit_pod_result(
            args.ledger,
            session_id,
            state["results"][pod_id],
            format_versions=format_versions,
            cursor=cursor,
        )
    entries = _entries_from_state(
        state, format_versions=format_versions, session_id=session_id
    )
    sync_report = sync_snapshot(
        args.ledger,
        entries,
        format_versions=format_versions,
        parameter_version=spec.parameter_version,
        scale_factor=spec.scale_factor,
    )
    state["scheduling_mode"] = mode_used
    state["rescored_pods"] = list(affected)
    state["invalidations"] = (
        invalidation.to_dict() if invalidation is not None else None
    )
    state["ledger_snapshot"] = sync_report.to_dict()
    atomic_write_text(args.state, canonical_json(state) + "\n")
    return {
        "ok": True,
        "mode": mode_used,
        "parameter_version": spec.parameter_version,
        "scale_factor": spec.scale_factor,
        "format_version_set": format_versions,
        "affected_pods": list(affected),
        "affected_nodes": (
            list(invalidation.affected_nodes) if invalidation is not None else sorted(spec.node_by_name)
        ),
        "invalidation_steps": (
            [dict(step) for step in invalidation.steps]
            if invalidation is not None
            else []
        ),
        "bindings": {
            pod_id: {
                "node_name": result["node_name"],
                "status": result["status"],
                "veto_code": result["veto_code"],
                "total_score": result["total_score"],
                "conclusion_digest": result["conclusion_digest"],
            }
            for pod_id, result in sorted(state["results"].items())
        },
        "ledger_snapshot": sync_report.to_dict(),
        "state_path": str(args.state),
    }


def cmd_verify_incremental(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    if not Path(args.state).exists():
        raise VerifyError(
            "previous scheduler state is required for incremental verification",
            node_name="-",
            details={"field": "state_path", "path": str(args.state)},
        )
    previous = read_json(args.state)
    invalidation = compute_invalidations(previous, spec)
    with ScoringManager(spec) as manager:
        incremental = schedule_incremental(
            spec, manager, previous, invalidation.affected_pods
        )
        full = schedule_full(spec, manager)
    difference = compare_results_stable(incremental, full)
    if difference is not None:
        pod_id, node_name, field = difference
        raise VerifyError(
            "incremental and full recomputation differ",
            pod_id=pod_id,
            node_name=node_name,
            details={"field": field},
        )
    for pod_id in sorted(set(previous.get("results", {})) - set(invalidation.affected_pods)):
        if pod_id not in incremental["results"]:
            continue
        if result_conclusion_bytes(previous["results"][pod_id]) != result_conclusion_bytes(
            incremental["results"][pod_id]
        ):
            raise VerifyError(
                "an unaffected pod conclusion changed bytes",
                pod_id=pod_id,
                node_name=str(incremental["results"][pod_id].get("node_name", "-")),
                details={"field": "stable_conclusion"},
            )
    return {
        "ok": True,
        "consistent": True,
        "affected_pods": list(invalidation.affected_pods),
        "affected_nodes": list(invalidation.affected_nodes),
        "unaffected_pods_byte_identical": True,
        "pod_count": len(spec.pods),
    }


def cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    return verify_ledger(args.ledger, spec, args.records)


def cmd_ledger_status(args: argparse.Namespace) -> dict[str, Any]:
    if not Path(args.ledger).exists():
        raise SnapshotError(
            "ledger file does not exist",
            details={"path": str(args.ledger)},
        )
    return {"ok": True, **ledger_status(args.ledger)}


def cmd_ledger_recover(args: argparse.Namespace) -> dict[str, Any]:
    if not Path(args.ledger).exists():
        raise SnapshotError(
            "ledger file does not exist",
            details={"path": str(args.ledger)},
        )
    return {"ok": True, **recover_ledger(args.ledger)}


def cmd_ledger_sync(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_spec(args.spec)
    batch = load_records(args.records, params=NormalizationParams.from_spec(spec))
    state = read_json(args.state)
    entries = _entries_from_state(state, format_versions=list(batch.format_versions))
    report = sync_snapshot(
        args.ledger,
        entries,
        format_versions=list(batch.format_versions),
        parameter_version=spec.parameter_version,
        scale_factor=spec.scale_factor,
        inject_digest_mismatch=args.inject_digest,
    )
    return {"ok": True, **report.to_dict()}


def cmd_inject_fault(args: argparse.Namespace) -> dict[str, Any]:
    if args.kind in {"record-corruption", "v1-width"}:
        batch = _load_records_with_injection(
            args.records, spec_path=args.spec, inject=args.kind
        )
        return {"ok": True, "injected": args.kind, "records": len(batch.records)}
    if args.kind == "ledger-digest":
        spec = load_spec(args.spec)
        batch = load_records(args.records, params=NormalizationParams.from_spec(spec))
        state = read_json(args.state)
        entries = _entries_from_state(state, format_versions=list(batch.format_versions))
        report = sync_snapshot(
            args.ledger,
            entries,
            format_versions=list(batch.format_versions),
            parameter_version=spec.parameter_version,
            scale_factor=spec.scale_factor,
            inject_digest_mismatch=True,
        )
        return {"ok": True, "injected": args.kind, **report.to_dict()}
    raise ValueError(f"unknown fault kind {args.kind!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="affinity-scheduler")
    parser.add_argument("--log-file", default="logs/structured.jsonl")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("generate-sample")
    sample.add_argument("--root", default=".")
    sample.add_argument("--port-base", type=int)
    sample.set_defaults(func=cmd_generate_sample)

    validate = sub.add_parser("validate-spec")
    validate.add_argument("--spec", default="config/spec.json")
    validate.set_defaults(func=cmd_validate_spec)

    normalize = sub.add_parser("normalize-records")
    normalize.add_argument("--records", default="data/records.jsonl")
    normalize.add_argument("--spec", default="config/spec.json")
    normalize.add_argument(
        "--inject", choices=["none", "record-corruption", "v1-width"], default="none"
    )
    normalize.set_defaults(func=cmd_normalize_records)

    migrate = sub.add_parser("migrate")
    migrate.add_argument("--records", default="data/records.jsonl")
    migrate.add_argument("--spec", default="config/spec.json")
    migrate.add_argument("--db", default="data/ledger.sqlite3")
    migrate.add_argument("--backup")
    migrate.add_argument("--batch-size", type=int, default=2)
    migrate.add_argument(
        "--abort-after-batches",
        type=int,
        help="test-only abrupt termination after N committed batches",
    )
    migrate.set_defaults(func=cmd_migrate)

    schedule = sub.add_parser("schedule")
    schedule.add_argument("--spec", default="config/spec.json")
    schedule.add_argument("--records", default="data/records.jsonl")
    schedule.add_argument("--state", default="data/state.json")
    schedule.add_argument("--ledger", default="data/ledger.sqlite3")
    schedule.add_argument("--mode", choices=["incremental", "full"], default="incremental")
    schedule.set_defaults(func=cmd_schedule)

    verify_inc = sub.add_parser("verify-incremental")
    verify_inc.add_argument("--spec", default="config/spec-v2.json")
    verify_inc.add_argument("--state", default="data/state.json")
    verify_inc.set_defaults(func=cmd_verify_incremental)

    verify = sub.add_parser("verify")
    verify.add_argument("--ledger", default="data/ledger.sqlite3")
    verify.add_argument("--spec", default="config/spec.json")
    verify.add_argument("--records", default="data/records.jsonl")
    verify.set_defaults(func=cmd_verify)

    status = sub.add_parser("ledger-status")
    status.add_argument("--ledger", default="data/ledger.sqlite3")
    status.set_defaults(func=cmd_ledger_status)

    recover = sub.add_parser("ledger-recover")
    recover.add_argument("--ledger", default="data/ledger.sqlite3")
    recover.set_defaults(func=cmd_ledger_recover)

    sync = sub.add_parser("ledger-sync")
    sync.add_argument("--ledger", default="data/ledger.sqlite3")
    sync.add_argument("--spec", default="config/spec.json")
    sync.add_argument("--records", default="data/records.jsonl")
    sync.add_argument("--state", default="data/state.json")
    sync.add_argument("--inject-digest", action="store_true")
    sync.set_defaults(func=cmd_ledger_sync)

    inject = sub.add_parser("inject-fault")
    inject.add_argument(
        "--kind", choices=["record-corruption", "v1-width", "ledger-digest"], required=True
    )
    inject.add_argument("--records", default="data/records.jsonl")
    inject.add_argument("--spec", default="config/spec.json")
    inject.add_argument("--ledger", default="data/ledger.sqlite3")
    inject.add_argument("--state", default="data/state.json")
    inject.set_defaults(func=cmd_inject_fault)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = StructuredLogger(args.log_file)
    try:
        result = args.func(args)
        logger.emit("OK", f"command {args.command} completed", command=args.command)
        _json_print(result)
        return 0
    except SchedulerError as exc:
        log_context = exc.as_dict()
        log_context.pop("code", None)
        log_context.pop("message", None)
        logger.emit(
            exc.code,
            exc.message,
            command=args.command,
            **log_context,
        )
        _json_print(exc.as_dict())
        return 2
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        logger.emit(
            "E_INTERNAL",
            str(exc),
            command=args.command,
            exception_type=type(exc).__name__,
        )
        _json_print(
            {
                "ok": False,
                "code": "E_INTERNAL",
                "message": str(exc),
                "exception_type": type(exc).__name__,
            }
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
