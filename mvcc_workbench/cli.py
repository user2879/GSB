from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from .errors import DiagnosticError, TaskExecutionError
from .ledger import Ledger
from .loader import load_dataset
from .logging_utils import EventLogger
from .renderer import render_full_report
from .workbench import RunOptions, Workbench


ERROR_EXIT_CODES = {
    "E_TXN": 2,
    "E_CHAIN": 3,
    "E_QUERY": 4,
    "E_REPLAY": 5,
    "E_SNAPSHOT": 6,
    "E_UI": 7,
    "E_TASK": 8,
}


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _emit_report(report: dict[str, Any], *, as_json: bool, dataset: Any) -> None:
    if as_json:
        _print_json(report)
    else:
        print(render_full_report(report, dataset), end="")


def _emit_error(error: DiagnosticError, *, as_json: bool, logger: EventLogger) -> int:
    logger.error(error)
    if as_json:
        _print_json(error.as_dict())
    else:
        print(json.dumps(error.as_dict(), ensure_ascii=False, sort_keys=True))
    return ERROR_EXIT_CODES.get(error.code, 1)


def _add_common(parser: argparse.ArgumentParser, *, database_required: bool = True) -> None:
    parser.add_argument("--input", required=True, type=Path, help="local dataset JSON")
    parser.add_argument("--db", required=database_required, type=Path, help="SQLite ledger path")
    parser.add_argument("--log", type=Path, help="structured JSONL event log")
    parser.add_argument("--json", action="store_true", help="emit one JSON object")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mvcc_workbench",
        description="Local deterministic MVCC visibility diagnostics workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="execute or resume deterministic visibility rounds")
    _add_common(run)
    run.add_argument("--crash-in-round", type=int, help="fault injection: exit after a round task is submitted")
    run.add_argument("--crash-after-submit", type=int, default=0)

    recover = subparsers.add_parser("recover", help="inspect crash-safe recovery state without generating a report")
    _add_common(recover)

    inject = subparsers.add_parser("inject", help="run one of the pre-generated exception samples")
    inject.add_argument("--kind", required=True, choices=("chain_cycle", "txn_commit_regress", "snapshot_digest"))
    inject.add_argument("--input", required=True, type=Path)
    inject.add_argument("--db", type=Path)
    inject.add_argument("--log", type=Path)
    inject.add_argument("--json", action="store_true")

    ui = subparsers.add_parser("ui", help="open the Tkinter diagnostic workbench")
    _add_common(ui)
    ui.add_argument("--run-before-open", action="store_true")

    return parser


def _run(args: argparse.Namespace) -> int:
    logger = EventLogger(args.log)
    try:
        dataset = load_dataset(args.input)
        logger.event(
            "dataset_loaded",
            input=str(args.input),
            manifest=dataset.manifest.as_dict(),
            metadata=dataset.metadata_digest(),
        )
        with Ledger(args.db) as ledger:
            workbench = Workbench(dataset, ledger)
            report = workbench.execute(
                RunOptions(
                    crash_in_round=args.crash_in_round,
                    crash_after_submit=args.crash_after_submit,
                )
            )
            for round_row in report.get("rounds", []):
                logger.event(
                    "round_committed",
                    generation=round_row["generation"],
                    round_no=round_row["round_no"],
                    ledger_cursor=round_row["ledger_cursor"],
                    content_digest=round_row["content_digest"],
                )
            if report.get("diagnostic"):
                logger.event(
                    "replay_reconciliation",
                    code=report["diagnostic"]["code"],
                    location=report["diagnostic"].get("location"),
                    old_snapshot_content_digest=report["diagnostic"].get("old_snapshot_content_digest"),
                    new_dataset_digest=report["diagnostic"].get("new_dataset_digest"),
                    affected_query_ids=(report.get("affected") or {}).get("affected_query_ids", []),
                )
            logger.event(
                "run_complete",
                status=report["status"],
                code=report["code"],
                generation=report["generation"],
                ledger_version=report["ledger_version"],
                conclusion_digest=report["conclusion_digest"],
                deferred_queries=report["deferred_queries"],
                affected=(report.get("affected") or {}).get("affected_query_ids", []),
            )
            _emit_report(report, as_json=args.json, dataset=dataset)
            return 0
    except DiagnosticError as error:
        return _emit_error(error, as_json=args.json, logger=logger)
    except TaskExecutionError as error:
        diagnostic = DiagnosticError(
            "E_TASK",
            {"read_point_id": "<worker>", "row_key": "<worker>", "field": "execution"},
            str(error),
        )
        return _emit_error(diagnostic, as_json=args.json, logger=logger)


def _recover(args: argparse.Namespace) -> int:
    logger = EventLogger(args.log)
    try:
        dataset = load_dataset(args.input)
        with Ledger(args.db) as ledger:
            generation = ledger.current_generation()
            snapshot = ledger.latest_snapshot()
            rounds = ledger.completed_rounds(generation) if generation is not None else []
            unknown = ledger.unknown_tasks()
            result = {
                "ok": True,
                "status": "RECOVERY_REQUIRED" if unknown else "CLEAN_RECOVERY",
                "code": "OK",
                "dataset_id": dataset.manifest.dataset_id,
                "generation": generation,
                "ledger_version": {
                    "schema_version": 1,
                    "ledger_cursor": ledger.ledger_seq(),
                    "generation": generation,
                },
                "completed_rounds": [
                    {
                        "generation": row["generation"],
                        "round_no": row["round_no"],
                        "ledger_cursor": row["ledger_cursor"],
                        "content_digest": row["content_digest"],
                    }
                    for row in rounds
                ],
                "committed_conclusion_count": sum(len(row["decisions"]) for row in rounds),
                "snapshot": None
                if snapshot is None
                else {
                    "generation": snapshot.get("generation"),
                    "ledger_cursor": snapshot.get("ledger_cursor"),
                    "content_digest": snapshot.get("content_digest"),
                    "conclusion_digest": snapshot.get("conclusion_digest"),
                    "cursor": snapshot.get("cursor"),
                },
                "unknown_tasks": unknown,
                "report_available": snapshot is not None and not unknown,
                "next_action": "RERUN_UNKNOWN_TASKS" if unknown else "RUN_OR_REPLAY",
            }
            logger.event(
                "recovery_inspected",
                status=result["status"],
                unknown_count=len(unknown),
                report_available=result["report_available"],
                ledger_version=result["ledger_version"],
            )
            if args.json:
                _print_json(result)
            else:
                print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
    except DiagnosticError as error:
        return _emit_error(error, as_json=args.json, logger=logger)


def _scratch_database(requested: Path | None) -> Path:
    if requested is None:
        return Path(tempfile.mkdtemp(prefix="mvcc-inject-")) / "ledger.sqlite"
    candidate = requested
    index = 1
    while candidate.exists():
        suffix = ".inject" if index == 1 else f".inject.{index}"
        candidate = requested.with_name(requested.stem + suffix + requested.suffix)
        index += 1
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _inject(args: argparse.Namespace) -> int:
    logger = EventLogger(args.log)
    try:
        dataset = load_dataset(args.input)
        if args.kind in ("chain_cycle", "txn_commit_regress"):
            raise AssertionError("loader should have rejected the injected sample")

        raw = json.loads(args.input.read_text(encoding="utf-8"))
        injection = raw.get("injection") or {}
        injected_digest = injection.get("snapshot_content_digest")
        if not isinstance(injected_digest, str):
            raise DiagnosticError(
                "E_SNAPSHOT",
                {"read_point_id": "<inject>", "row_key": "<inject>", "field": "injection.snapshot_content_digest"},
                "snapshot injection sample does not contain a digest",
            )
        db_path = _scratch_database(args.db)
        with Ledger(db_path) as ledger:
            Workbench(dataset, ledger).execute()
        connection = sqlite3.connect(db_path)
        try:
            connection.execute(
                "UPDATE snapshots SET content_digest = ? WHERE snapshot_no = (SELECT MAX(snapshot_no) FROM snapshots)",
                (injected_digest,),
            )
            connection.commit()
        finally:
            connection.close()
        logger.event(
            "snapshot_tampered",
            db=str(db_path),
            injected_snapshot_content_digest=injected_digest,
        )
        with Ledger(db_path) as ledger:
            ledger.verify_integrity()
        raise AssertionError("tampered ledger integrity verification unexpectedly passed")
    except DiagnosticError as error:
        logger.event("injection_result", kind=args.kind, code=error.code, location=error.location)
        return _emit_error(error, as_json=args.json, logger=logger)


def _ui(args: argparse.Namespace) -> int:
    logger = EventLogger(args.log)
    try:
        dataset = load_dataset(args.input)
        with Ledger(args.db) as ledger:
            workbench = Workbench(dataset, ledger)
            report = workbench.execute() if args.run_before_open else workbench.refresh_view()
            try:
                from .ui import UiUnavailable, launch_ui

                launch_ui(workbench, report)
            except UiUnavailable as error:
                diagnostic = DiagnosticError(
                    "E_UI",
                    {"read_point_id": "<ui>", "row_key": "<ui>", "field": "tkinter"},
                    str(error),
                )
                logger.error(diagnostic)
                if args.json:
                    _print_json(diagnostic.as_dict())
                else:
                    print(json.dumps(diagnostic.as_dict(), ensure_ascii=False, sort_keys=True))
                    print(render_full_report(report, dataset), end="")
                return ERROR_EXIT_CODES["E_UI"]
            logger.event("ui_opened", input=str(args.input), db=str(args.db))
            return 0
    except DiagnosticError as error:
        return _emit_error(error, as_json=args.json, logger=logger)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "recover":
        return _recover(args)
    if args.command == "inject":
        return _inject(args)
    if args.command == "ui":
        return _ui(args)
    parser.error(f"unknown command {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
