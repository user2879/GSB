"""Command-line interface for the local materialized-view component."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, NoReturn, Sequence

from .canonical import canonical_json
from .changes import load_changes
from .engine import MaterializationEngine
from .errors import ColmvError
from .evolution import apply_evolution, build_invalidation_plan
from .ledger import LedgerStore
from .replay import (
    config_from_ledger,
    replay_and_verify,
    split_changes,
    split_uniform,
    verify_paths,
)
from .schema import load_config


class CliParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # type: ignore[override]
        raise ColmvError("E_USAGE", message, field="arguments")


def _add_injection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--inject",
        choices=("none", "truncate-page", "zero-budget"),
        default="none",
        help="deterministic failure injection: truncate the first page or force budget zero",
    )


def _add_batch_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--batch-size", type=int, default=4, help="uniform changes per batch")
    group.add_argument(
        "--batch-plan",
        type=str,
        default=None,
        help="comma-separated batch sizes, e.g. 3,6,3,6,2; must consume the whole stream",
    )


def build_parser() -> CliParser:
    parser = CliParser(prog="python -m colmv", description=__doc__)
    parser.add_argument("--version", action="version", version="colmv 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check Python, NumPy, and SQLite prerequisites")
    doctor.add_argument("--output")

    validate = subparsers.add_parser("validate", help="strictly validate catalog and view definitions")
    validate.add_argument("--catalog", required=True)
    validate.add_argument("--views", required=True)
    validate.add_argument("--output")
    _add_injection(validate)

    run = subparsers.add_parser("run", help="apply an NDJSON change stream and persist an SQLite ledger")
    run.add_argument("--catalog", required=True)
    run.add_argument("--views", required=True)
    run.add_argument("--changes", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--budget", type=int, required=True)
    run.add_argument("--path", choices=("auto", "resident", "rescan"), default="auto")
    run.add_argument("--resume", action="store_true", help="verify committed batches and continue from the ledger cursor")
    run.add_argument(
        "--crash-during-batch",
        type=int,
        default=None,
        help="after recording the batch intent, force-exit before processing this 1-based batch",
    )
    run.add_argument(
        "--crash-after-commits",
        type=int,
        default=None,
        help="force-exit immediately after this many whole batches have committed",
    )
    run.add_argument("--output")
    _add_batch_options(run)
    _add_injection(run)

    verify = subparsers.add_parser("verify-paths", help="run resident and rescan paths and compare result bytes")
    verify.add_argument("--catalog", required=True)
    verify.add_argument("--views", required=True)
    verify.add_argument("--changes", required=True)
    verify.add_argument("--budget", type=int, required=True)
    verify.add_argument("--output")
    _add_batch_options(verify)
    _add_injection(verify)

    replay = subparsers.add_parser("replay", help="replay committed ledger batches and verify every field")
    replay.add_argument("--catalog", required=True)
    replay.add_argument("--ledger", required=True)
    replay.add_argument("--changes", required=True)
    replay.add_argument("--scheme-version", type=int, default=None)
    replay.add_argument("--output")

    recover = subparsers.add_parser("recover", help="mark uncommitted batch groups unknown and report resume cursor")
    recover.add_argument("--ledger", required=True)
    recover.add_argument("--output")

    snapshot = subparsers.add_parser("snapshot-status", help="verify ledger digest chain and session snapshot")
    snapshot.add_argument("--ledger", required=True)
    snapshot.add_argument("--output")

    evolve = subparsers.add_parser("evolve", help="invalidate references and whole-replace the current scheme")
    evolve.add_argument("--catalog", required=True)
    evolve.add_argument("--old-views", required=True)
    evolve.add_argument("--new-views", required=True)
    evolve.add_argument("--old-budget", type=int, required=True)
    evolve.add_argument("--new-budget", type=int, required=True)
    evolve.add_argument("--changes", default=None)
    evolve.add_argument("--ledger", default=None)
    evolve.add_argument("--output")
    _add_injection(evolve)

    return parser


def _emit(payload: dict[str, Any], output: str | None = None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(rendered)


def _load_config(args: argparse.Namespace):
    return load_config(
        args.catalog,
        args.views,
        inject_truncate=getattr(args, "inject", "none") == "truncate-page",
    )


def _effective_budget(args: argparse.Namespace) -> int:
    if getattr(args, "inject", "none") == "zero-budget":
        return 0
    return int(args.budget)


def _require_positive_budget(budget: int) -> None:
    if budget <= 0:
        raise ColmvError(
            "E_BUDGET",
            "integer slot budget must be greater than zero",
            field="budget",
            details={"budget": budget, "required_minimum": 1},
        )


def _parse_batch_sizes(raw: str) -> list[int]:
    try:
        sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ColmvError("E_USAGE", f"invalid --batch-plan: {exc}", field="batch_plan") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise ColmvError("E_USAGE", "--batch-plan must contain positive integers", field="batch_plan")
    return sizes


def _full_batches(changes: Sequence[Any], args: argparse.Namespace) -> list[list[Any]]:
    if getattr(args, "batch_plan", None):
        return split_changes(changes, _parse_batch_sizes(args.batch_plan))
    return split_uniform(changes, int(args.batch_size))


def _run_batches_committed(
    ledger: LedgerStore,
    engine: MaterializationEngine,
    *,
    scheme_version: int,
    batches: Sequence[Sequence[Any]],
    start_batch_no: int,
    crash_during_batch: int | None,
    crash_after_commits: int | None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    committed_this_call = 0
    for offset, batch_changes in enumerate(batches):
        batch_no = start_batch_no + offset
        decision = engine.decide_batch(batch_changes)
        planned_groups = engine.preview_touched_groups(batch_changes)
        attempt_id = ledger.begin_batch_attempt(
            scheme_version=scheme_version,
            batch_no=batch_no,
            start_seq=batch_changes[0].seq,
            end_seq=batch_changes[-1].seq,
            path=decision.path,
            planned_groups=[ref.to_dict() for ref in planned_groups],
        )
        if crash_during_batch is not None and batch_no == crash_during_batch:
            os._exit(137)
        batch_result = engine.process_batch(batch_changes, batch_no=batch_no)
        commit_info = ledger.commit_batch(
            attempt_id=attempt_id,
            scheme_version=scheme_version,
            batch_result=batch_result,
            materialized_result=engine.materialized_result(),
        )
        committed_this_call += 1
        summaries.append(
            {
                "batch_no": batch_no,
                "start_seq": batch_result["start_seq"],
                "end_seq": batch_result["end_seq"],
                "change_count": batch_result["change_count"],
                "path": batch_result["path"],
                "decision": batch_result["decision"],
                "hits": batch_result["hits"],
                "replacement_event_count": len(batch_result["replacement_events"]),
                "rescan_event_count": len(batch_result["rescan_events"]),
                "rescan_cost": batch_result["rescan_cost"],
                "slot_partition": batch_result["slot_partition"],
                "affected_pages": batch_result["affected_pages"],
                "affected_groups": batch_result["affected_groups"],
                "events": batch_result["events"],
                "commit": commit_info,
            }
        )
        if crash_after_commits is not None and committed_this_call == crash_after_commits:
            os._exit(137)
    return summaries


def _run_output(
    *,
    config,
    engine: MaterializationEngine,
    scheme_version: int,
    budget: int,
    batch_summaries: list[dict[str, Any]],
    recovery: dict[str, Any],
    snapshot: dict[str, Any],
    replay_report: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "command": "run",
        "scheme": {
            "version": scheme_version,
            "fingerprint": config.scheme_fingerprint(budget),
            "budget": budget,
            "scale_power": config.scale_power,
            "path_mode": engine.path_mode,
        },
        "recovery": recovery,
        "snapshot": snapshot,
        "replay": replay_report,
        "batches": batch_summaries,
        "switch_history": engine.switch_history,
        "batch_decisions": engine.batch_decisions,
        "slot_summary": {
            "budget": budget,
            "hits": engine.total_hits,
            "replacements": engine.total_replacements,
            "rescans": engine.total_rescans,
            "resident_group_count": len(engine.cache.snapshot()),
            "final_resident_groups": [ref.to_dict() for ref in engine.cache.snapshot()],
            "complement_checks": [
                {
                    "batch_no": batch["batch_no"],
                    "complete_for_affected": batch["slot_partition"]["complete_for_affected"],
                    "disjoint": batch["slot_partition"]["disjoint"],
                }
                for batch in batch_summaries
            ],
        },
        "materialized_digest": engine.materialized_digest(),
        "materialized_byte_length": len(engine.materialized_bytes()),
        "materialized": engine.materialized_result(),
    }


def command_doctor(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ColmvError("E_USAGE", f"NumPy is not installed: {exc}", field="numpy") from exc
    numpy_ok = tuple(int(part) for part in np.__version__.split(".")[:2]) >= (1, 24)
    python_ok = sys.version_info >= (3, 11)
    payload = {
        "ok": python_ok and numpy_ok,
        "command": "doctor",
        "python": sys.version.split()[0],
        "python_requirement": ">=3.11",
        "python_ok": python_ok,
        "numpy": np.__version__,
        "numpy_requirement": ">=1.24",
        "numpy_ok": numpy_ok,
        "sqlite": sqlite3.sqlite_version,
        "sqlite_ok": True,
    }
    if not python_ok:
        raise ColmvError("E_USAGE", "Python 3.11 or newer is required", field="python")
    if not numpy_ok:
        raise ColmvError("E_USAGE", "NumPy 1.24 or newer is required", field="numpy")
    return payload


def command_validate(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args)
    return {
        "ok": True,
        "command": "validate",
        "base": {
            "name": config.table.name,
            "row_count": config.table.row_count,
            "row_key": config.table.row_key,
            "columns": [column.__dict__ for column in config.table.columns],
            "page_count": len(config.table.pages),
            "pages": [
                {
                    "page_id": page.page_id,
                    "file": page.file,
                    "start_row": page.start_row,
                    "end_row": page.end_row,
                    "row_count": page.row_count,
                }
                for page in config.table.pages
            ],
        },
        "scale_power": config.scale_power,
        "views": [view.to_dict() for view in config.views],
        "catalog_digest": config.catalog_digest,
        "views_digest": config.views_digest,
    }


def command_run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args)
    budget = _effective_budget(args)
    _require_positive_budget(budget)
    changes = load_changes(args.changes, config.table)
    all_batches = _full_batches(changes, args)

    with LedgerStore(args.ledger) as ledger:
        recovery = ledger.recover_uncommitted()
        snapshot_check = ledger.verify_snapshot()
        expected_fingerprint = config.scheme_fingerprint(budget)
        current = ledger.current_scheme()

        if args.resume:
            if current is not None and current["fingerprint"] != expected_fingerprint:
                raise ColmvError(
                    "E_LEDGER",
                    "resume scheme fingerprint does not match the current ledger scheme; run evolve or start a new ledger",
                    field="scheme_fingerprint",
                    details={
                        "ledger_fingerprint": current["fingerprint"],
                        "requested_fingerprint": expected_fingerprint,
                    },
                )
            if current is None:
                scheme_version = ledger.register_scheme(config, budget=budget)
                engine = MaterializationEngine(
                    config,
                    budget=budget,
                    path_mode=args.path,
                    scheme_version=scheme_version,
                )
                replay_report = None
                remaining = list(changes)
                next_batch_no = 1
            else:
                scheme_version = int(current["scheme_version"])
                committed = ledger.committed_batches(scheme_version)
                if committed:
                    engine, replay_report = replay_and_verify(
                        ledger,
                        catalog_path=args.catalog,
                        changes=changes,
                        scheme_version=scheme_version,
                    )
                    # Replay verifies history; resume may use its own path mode
                    # only for new batches, but the ledger scheme remains fixed.
                    engine.path_mode = args.path
                    remaining = [
                        change
                        for change in changes
                        if engine.last_seq is None or change.seq > engine.last_seq
                    ]
                    next_batch_no = int(committed[-1]["batch_no"]) + 1
                else:
                    engine = MaterializationEngine(
                        config,
                        budget=budget,
                        path_mode=args.path,
                        scheme_version=scheme_version,
                    )
                    replay_report = None
                    remaining = list(changes)
                    next_batch_no = 1
            if remaining:
                if args.batch_plan:
                    planned = split_changes(remaining, _parse_batch_sizes(args.batch_plan))
                else:
                    planned = split_uniform(remaining, int(args.batch_size))
            else:
                planned = []
        else:
            scheme_version = ledger.register_scheme(config, budget=budget)
            existing_batches = ledger.committed_batches(scheme_version)
            if existing_batches:
                raise ColmvError(
                    "E_LEDGER",
                    "the current scheme already has committed batches; use --resume or a new ledger path",
                    field="ledger",
                    details={
                        "scheme_version": scheme_version,
                        "last_batch": existing_batches[-1]["batch_no"],
                        "last_cursor": existing_batches[-1]["cursor"],
                    },
                )
            engine = MaterializationEngine(
                config,
                budget=budget,
                path_mode=args.path,
                scheme_version=scheme_version,
            )
            replay_report = None
            planned = all_batches
            next_batch_no = 1

        summaries = _run_batches_committed(
            ledger,
            engine,
            scheme_version=scheme_version,
            batches=planned,
            start_batch_no=next_batch_no,
            crash_during_batch=args.crash_during_batch,
            crash_after_commits=args.crash_after_commits,
        )
        output = _run_output(
            config=config,
            engine=engine,
            scheme_version=scheme_version,
            budget=budget,
            batch_summaries=summaries,
            recovery=recovery,
            snapshot={**snapshot_check, "current_after_run": ledger.verify_snapshot()},
            replay_report=replay_report,
        )
        output["ledger"] = {
            "path": str(Path(args.ledger)),
            "format_version": ledger.ledger_meta().get("ledger_format_version"),
            "meta": ledger.ledger_meta(),
            "current_scheme": ledger.current_scheme(),
            "content_chain": ledger.verify_content_chain(),
            "last_committed_batch": ledger.last_committed_batch(),
        }
        return output


def command_verify_paths(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args)
    budget = _effective_budget(args)
    changes = load_changes(args.changes, config.table)
    batch_sizes = [len(batch) for batch in _full_batches(changes, args)]
    result = verify_paths(
        config,
        changes,
        budget=budget,
        batch_sizes=batch_sizes,
    )
    result["command"] = "verify-paths"
    return result


def command_replay(args: argparse.Namespace) -> dict[str, Any]:
    with LedgerStore(args.ledger) as ledger:
        recovery = ledger.recover_uncommitted()
        snapshot = ledger.verify_snapshot()
        scheme = ledger.current_scheme() if args.scheme_version is None else ledger.scheme(args.scheme_version)
        if scheme is None:
            raise ColmvError("E_REPLAY", "ledger has no scheme to replay", field="scheme_version")
        config = config_from_ledger(ledger, args.catalog, int(scheme["scheme_version"]))
        changes = load_changes(args.changes, config.table)
        _, report = replay_and_verify(
            ledger,
            catalog_path=args.catalog,
            changes=changes,
            scheme_version=int(scheme["scheme_version"]),
        )
        report["command"] = "replay"
        report["recovery"] = recovery
        report["snapshot"] = snapshot
        report["ledger"] = {
            "path": str(Path(args.ledger)),
            "meta": ledger.ledger_meta(),
            "content_chain": ledger.verify_content_chain(),
        }
        return report


def command_recover(args: argparse.Namespace) -> dict[str, Any]:
    with LedgerStore(args.ledger) as ledger:
        recovery = ledger.recover_uncommitted()
        snapshot = ledger.verify_snapshot()
        current = ledger.current_scheme()
        unknown = ledger.unknown_groups()
        return {
            "ok": True,
            "command": "recover",
            "recovery": recovery,
            "unknown_group_count": len(unknown),
            "unknown_groups": unknown,
            "exit_report_available": False,
            "snapshot": snapshot,
            "current_scheme": current,
            "last_committed_batch": ledger.last_committed_batch(),
            "content_chain": ledger.verify_content_chain(),
            "meta": ledger.ledger_meta(),
        }


def command_snapshot_status(args: argparse.Namespace) -> dict[str, Any]:
    with LedgerStore(args.ledger) as ledger:
        return {
            "ok": True,
            "command": "snapshot-status",
            "snapshot_check": ledger.verify_snapshot(),
            "snapshot": ledger.snapshot(),
            "content_chain": ledger.verify_content_chain(),
            "current_scheme": ledger.current_scheme(),
            "last_committed_batch": ledger.last_committed_batch(),
            "meta": ledger.ledger_meta(),
        }


def command_evolve(args: argparse.Namespace) -> dict[str, Any]:
    inject_truncate = args.inject == "truncate-page"
    old_config = load_config(args.catalog, args.old_views, inject_truncate=inject_truncate)
    new_config = load_config(args.catalog, args.new_views, inject_truncate=inject_truncate)
    old_budget = 0 if args.inject == "zero-budget" else int(args.old_budget)
    new_budget = 0 if args.inject == "zero-budget" else int(args.new_budget)
    changes = load_changes(args.changes, new_config.table) if args.changes else []
    plan = build_invalidation_plan(
        old_config=old_config,
        new_config=new_config,
        old_budget=old_budget,
        new_budget=new_budget,
        changes=changes,
    )
    plan["ok"] = True
    plan["command"] = "evolve"
    if args.ledger:
        with LedgerStore(args.ledger) as ledger:
            ledger.recover_uncommitted()
            ledger.verify_snapshot()
            application = apply_evolution(
                ledger,
                new_config=new_config,
                new_budget=new_budget,
                plan=plan,
            )
            plan["ledger_application"] = application
            plan["current_scheme"] = ledger.current_scheme()
            plan["snapshot"] = ledger.verify_snapshot()
    else:
        changed = bool(plan["changed_views"] or plan["budget_changed"])
        plan["ledger_application"] = {
            "old_scheme_version": 1,
            "new_scheme_version": 2 if changed else 1,
            "scheme_version_changed": changed,
            "monotonic_increment": True,
            "old_traces_marked_stale": changed,
            "snapshot_action": "not_persisted_without_ledger",
            "current_conclusions_visible": not changed,
            "requires_recompute": changed,
        }
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        handlers = {
            "doctor": command_doctor,
            "validate": command_validate,
            "run": command_run,
            "verify-paths": command_verify_paths,
            "replay": command_replay,
            "recover": command_recover,
            "snapshot-status": command_snapshot_status,
            "evolve": command_evolve,
        }
        payload = handlers[args.command](args)
        _emit(payload, getattr(args, "output", None))
        return 0
    except ColmvError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        error = ColmvError("E_USAGE", "interrupted by user")
        print(json.dumps(error.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return error.exit_code
    except Exception as exc:  # pragma: no cover - last-resort CLI envelope
        error = ColmvError(
            "E_INTERNAL",
            f"unexpected internal error: {type(exc).__name__}: {exc}",
        )
        print(json.dumps(error.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return error.exit_code
