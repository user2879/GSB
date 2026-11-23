"""Command-line acceptance driver and exception injection entry points."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, canonical_text
from .engine import PlanEngine
from .errors import E_REF, E_SNAPSHOT, E_SYNTAX, AppError
from .logging_utils import StructuredLogger
from .snapshots import read_snapshot
from .spec import load_workspace


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE = ROOT / "samples" / "workspace.json"
DEFAULT_LEDGER = ROOT / "data" / "ledger.sqlite"
DEFAULT_SNAPSHOTS = ROOT / "snapshots"
DEFAULT_LOG = ROOT / "logs" / "app.jsonl"


def _remove_ledger(path: Path) -> None:
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        candidate.unlink(missing_ok=True)


def _json_error(error: AppError, log_path: Path) -> dict[str, Any]:
    result = error.to_dict()
    result.update(
        {
            "mono_ns": time.monotonic_ns(),
            "log_path": str(log_path),
            "business_code": error.code,
        }
    )
    return result


def _cmd_check(args: argparse.Namespace) -> int:
    workspace = load_workspace(args.workspace)
    payload = {
        "ok": True,
        "name": workspace.name,
        "workspace_id": workspace.workspace_id,
        "tables": [table.name for table in workspace.tables],
        "queries": [query.name for query in workspace.queries],
        "edit_batches": len(workspace.edit_batches),
        "table_digest": workspace.table_digest(),
        "sql_digest": workspace.sql_digest(),
        "scale_digest": workspace.scale_digest(),
        "spec_digest": workspace.spec_digest(),
        "budget_ms": workspace.interaction_budget_ms,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _new_engine(args: argparse.Namespace, *, budget_ms: float | None = None) -> PlanEngine:
    return PlanEngine(
        args.workspace,
        ledger_path=args.ledger,
        snapshot_dir=args.snapshots,
        log_path=args.log,
        budget_ms=budget_ms,
    )


def _cmd_demo(args: argparse.Namespace) -> int:
    ledger = Path(args.ledger)
    log_path = Path(args.log)
    snapshots = Path(args.snapshots)
    snapshot_path = Path(args.snapshot_path) if args.snapshot_path else snapshots / "session-snapshot.json"
    if args.reset_ledger:
        _remove_ledger(ledger)
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    workspace = load_workspace(args.workspace)
    engine = _new_engine(args, budget_ms=args.budget)
    report: dict[str, Any] = {
        "ok": True,
        "generated_mono_ns": time.monotonic_ns(),
        "workspace": workspace.name,
        "workspace_id": workspace.workspace_id,
        "ledger": str(ledger),
        "snapshot_dir": str(snapshots),
        "budget_ms": engine.budget_ms,
        "baseline": {
            "queries": sorted(engine.states),
            "ledger_cursor": engine.ledger.latest_cursor(engine.workspace_id),
            "recovery": engine.recovery,
        },
        "initial_queries": [],
        "edit_batches": [],
        "authoritative": None,
        "snapshot": None,
        "restart_recovery": None,
        "snapshot_restore": None,
        "ledger_truth": {},
    }

    for query in workspace.queries:
        quick = engine.quick_query(query.name)
        report["initial_queries"].append(quick)

    for batch in workspace.edit_batches:
        result = engine.submit_edits(list(batch.edits), query_name=batch.query, background_verify=False)
        verification = result["verification"]
        if verification.get("status") != "MATCH":
            report["ok"] = False
        report["edit_batches"].append(
            {
                "name": batch.name,
                "query": batch.query,
                "edits": list(batch.edits),
                "batch_id": result["batch_id"],
                "edit_seq_start": result["edit_seq_start"],
                "edit_seq_end": result["edit_seq_end"],
                "parameter_version": result["parameter_version"],
                "incremental_ms": result["incremental_ms"],
                "latency_code": result["latency_code"],
                "affected_nodes": result["effects"][0]["affected_nodes"] if len(result["effects"]) == 1 else sorted({node for effect in result["effects"] for node in effect["affected_nodes"]}),
                "unchanged_nodes": result["effects"][0]["unchanged_nodes"] if len(result["effects"]) == 1 else [],
                "propagation": [effect["propagation"] for effect in result["effects"]],
                "plan_summary": result["view"]["plan_summary"],
                "estimated_rows": result["view"]["estimated_rows"],
                "incremental_vs_full": verification,
                "diff": result["view"]["diff"],
                "ledger_stale_batches": engine.ledger.count_stale_batches(engine.workspace_id),
            }
        )

    active_query = args.query or workspace.queries[0].name
    quick = engine.quick_query(active_query)
    report["final_quick_query"] = quick
    authoritative = engine.authoritative_recompute(active_query)
    report["authoritative"] = authoritative
    report["ledger_replay"] = engine.replay_ledger()
    snapshot = engine.create_snapshot(active_query, snapshot_path)
    report["snapshot"] = {
        key: value for key, value in snapshot.items() if key != "document"
    }
    snapshot_document = read_snapshot(snapshot_path)

    committed_summary_before_crash = engine.current_view(active_query)["plan_summary"]
    crash_cursor = engine.ledger.next_cursor(engine.workspace_id)
    crash_attempt_id = engine.ledger.begin_attempt(
        workspace_id=engine.workspace_id,
        cursor=crash_cursor,
        query_name=active_query,
        edits=[
            {
                "type": "modify_aggregate",
                "target": "aggregate",
                "name": "total_amount",
                "function": "sum",
                "column": "amount",
            }
        ],
        parameter_version=engine.states[active_query].parameter_version + 1,
        next_edit_seq=engine.states[active_query].edit_seq + 1,
    )

    restarted = _new_engine(args, budget_ms=args.budget)
    report["restart_recovery"] = restarted.recovery
    report["restart_recovery"]["crash_simulation"] = {
        "attempt_id": crash_attempt_id,
        "uncommitted_cursor": crash_cursor,
        "committed_plan_summary_before_crash": committed_summary_before_crash,
        "restored_plan_summary": restarted.current_view(active_query)["plan_summary"],
        "uncommitted_edit_applied": restarted.current_view(active_query)["plan_summary"]
        != committed_summary_before_crash,
        "exit_report_claimed": False,
    }
    restored = restarted.restore_snapshot(snapshot_path)
    report["snapshot_restore"] = restored
    if restored.get("replay", {}).get("status") != "MATCH":
        report["ok"] = False

    report["ledger_truth"]["latencies"] = restarted.ledger.recent_latency(restarted.workspace_id, 30)
    report["ledger_truth"]["verifications"] = restarted.ledger.recent_verifications(restarted.workspace_id, 30)
    report["ledger_truth"]["latest_cursor"] = restarted.ledger.latest_cursor(restarted.workspace_id)
    report["ledger_truth"]["snapshot_content_digest"] = snapshot_document.get("content_digest")
    report["ledger_truth"]["truth_source"] = "SQLite ledger latency_records and verification_records"
    report["all_quick_queries_within_budget"] = all(
        item.get("within_budget", False)
        for item in report["initial_queries"] + [report.get("final_quick_query", {})]
        if item
    )
    report["summary"] = {
        "edit_batches": len(report["edit_batches"]),
        "all_incremental_full_match": all(
            item["incremental_vs_full"].get("status") == "MATCH" for item in report["edit_batches"]
        ),
        "ledger_replay_match": report["ledger_replay"].get("status") == "MATCH",
        "snapshot_restore_match": report["snapshot_restore"].get("replay", {}).get("status") == "MATCH",
        "final_plan_summary": engine.current_view(active_query)["plan_summary"],
        "ledger_cursor": restarted.ledger.latest_cursor(restarted.workspace_id),
    }
    report_path = Path(args.report)
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nverification report written to {report_path}", file=sys.stderr)
    return 0 if report["ok"] else 1


def _inject_syntax(engine: PlanEngine, query_name: str) -> AppError:
    original = engine.states[query_name].query.sql
    try:
        engine.update_query_sql(query_name, original + " WHERE")
    except AppError as error:
        return error
    raise AssertionError("syntax injection unexpectedly succeeded")


def _inject_ref(engine: PlanEngine, query_name: str) -> AppError:
    try:
        engine.submit_edits(
            [{"type": "prune_columns", "target": "missing_node_for_injection", "columns": ["does_not_exist"]}],
            query_name=query_name,
        )
    except AppError as error:
        return error
    raise AssertionError("reference injection unexpectedly succeeded")


def _inject_snapshot(engine: PlanEngine, query_name: str, snapshot_dir: Path) -> AppError:
    original = snapshot_dir / "injection-original.json"
    tampered = snapshot_dir / "injection-tampered.json"
    engine.create_snapshot(query_name, original)
    document = read_snapshot(original)
    document["content_digest"] = "0" * 64
    atomic_write_json(tampered, document)
    try:
        engine.restore_snapshot(tampered)
    except AppError as error:
        return error
    raise AssertionError("snapshot injection unexpectedly succeeded")


def _cmd_inject(args: argparse.Namespace) -> int:
    ledger = Path(args.ledger)
    log_path = Path(args.log)
    if args.reset_ledger:
        _remove_ledger(ledger)
    Path(args.snapshots).mkdir(parents=True, exist_ok=True)
    engine = _new_engine(args, budget_ms=args.budget)
    query_name = args.query or engine.workspace.queries[0].name
    expected_code = {
        "syntax": E_SYNTAX,
        "bad-ref": E_REF,
        "snapshot-digest": E_SNAPSHOT,
    }[args.kind]
    try:
        if args.kind == "syntax":
            error = _inject_syntax(engine, query_name)
        elif args.kind == "bad-ref":
            error = _inject_ref(engine, query_name)
        else:
            error = _inject_snapshot(engine, query_name, Path(args.snapshots))
    except AppError as premature:
        error = premature
        if error.code != expected_code:
            print(json.dumps(_json_error(error, log_path), ensure_ascii=False, indent=2))
            return 1
    payload = _json_error(error, log_path)
    if payload.get("edit_seq") is None or (args.kind == "snapshot-digest" and payload.get("edit_seq") == 0):
        try:
            payload["edit_seq"] = engine.next_edit_seq(query_name)
        except Exception:
            pass
    payload["expected_code"] = expected_code
    payload["injection"] = args.kind
    payload["ok"] = error.code == expected_code
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


def _cmd_ui(args: argparse.Namespace) -> int:
    try:
        from .ui import run_ui
    except Exception as error:
        print(json.dumps({"ok": False, "code": "E_TK", "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    if args.reset_ledger:
        _remove_ledger(Path(args.ledger))
    try:
        return run_ui(
            workspace_path=args.workspace,
            ledger_path=args.ledger,
            snapshot_dir=args.snapshots,
            log_path=args.log,
            budget_ms=args.budget,
        )
    except Exception as error:
        print(json.dumps({"ok": False, "code": "E_TK", "message": str(error)}, ensure_ascii=False, indent=2))
        return 2


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--budget", type=float, default=None, help="override interaction budget in milliseconds")
    parser.add_argument("--reset-ledger", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m planner_app",
        description="Deterministic local SQL logical-plan editor and verifier",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="load and strictly validate a workspace JSON")
    check.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    check.set_defaults(func=_cmd_check)

    demo = subparsers.add_parser("demo", help="run the complete local acceptance path")
    _add_common_paths(demo)
    demo.add_argument("--query", default=None)
    demo.add_argument("--snapshot-path", default=None)
    demo.add_argument("--report", default=str(ROOT / "artifacts" / "verification-report.json"))
    demo.set_defaults(func=_cmd_demo)

    inject = subparsers.add_parser("inject", help="inject one of the three required faults")
    _add_common_paths(inject)
    inject.add_argument("--kind", choices=("syntax", "bad-ref", "snapshot-digest"), required=True)
    inject.add_argument("--query", default=None)
    inject.set_defaults(func=_cmd_inject)

    ui = subparsers.add_parser("ui", help="launch the Tkinter editing application")
    _add_common_paths(ui)
    ui.set_defaults(func=_cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AppError as error:
        log_path = Path(getattr(args, "log", DEFAULT_LOG))
        StructuredLogger(log_path).error(error, "cli_error", edit_seq=error.edit_seq)
        payload = _json_error(error, log_path)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
