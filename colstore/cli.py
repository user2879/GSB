"""Command line entry point for the columnar scan system.

Examples
--------
python -m colstore.cli prepare
python -m colstore.cli scan --query query_and.json
python -m colstore.cli check --query query_eq.json --parallelism 4
python -m colstore.cli inject --kind payload_truncation
python -m colstore.cli inject --kind encoding_mismatch
python -m colstore.cli inject --kind zero_budget
python -m colstore.cli snapshot save --id s1
python -m colstore.cli snapshot open --id s1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from .engine import ScanEngine
from .errors import ColStoreError
from .model import BudgetParams, Encoding, SegmentDefinition
from .prepare_samples import DATA_DIR, SAMPLE_DIR, prepare
from .segment import (
    inject_encoding_mismatch,
    inject_payload_truncation,
    load_segment,
)
from .snapshot import SnapshotStore
from .structured_log import StructuredLogger

LOG_PATH = os.path.join(DATA_DIR, "structured.log.jsonl")
SNAPSHOT_DB = os.path.join(DATA_DIR, "sessions.db")


def _load_engine(budget: BudgetParams | None = None) -> ScanEngine:
    engine = ScanEngine(DATA_DIR, budget)
    for filename in sorted(os.listdir(SAMPLE_DIR)):
        if not filename.startswith("segment_") or not filename.endswith(".json"):
            continue
        with open(os.path.join(SAMPLE_DIR, filename), "r", encoding="utf-8") as handle:
            engine.register_definition(SegmentDefinition.from_dict(json.load(handle)))
    engine.build_segments()
    return engine


def _read_query(name: str) -> Dict[str, Any]:
    path = name if os.path.isabs(name) else os.path.join(SAMPLE_DIR, name)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _emit(payload: Any, code: int = 0) -> int:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return code


def cmd_prepare(_args: argparse.Namespace) -> int:
    return _emit({"prepared": prepare()})


def cmd_scan(args: argparse.Namespace) -> int:
    logger = StructuredLogger(LOG_PATH)
    budget = BudgetParams(
        max_parallelism=args.parallelism,
        memory_budget_bytes=args.memory_budget,
        per_segment_concurrency=args.segment_concurrency,
    )
    engine = _load_engine(budget)
    query = _read_query(args.query)
    logger.info("scan_started", query=args.query, budget=budget.to_dict(),
                budget_version=engine.budget_version)
    result = engine.scan(query)
    logger.info(
        "scan_completed",
        result_id=result.result_id,
        rows_matched=result.stats.rows_matched,
        peak_memory_bytes=result.stats.peak_memory_bytes,
    )
    return _emit(result.to_dict())


def cmd_check(args: argparse.Namespace) -> int:
    logger = StructuredLogger(LOG_PATH)
    engine = _load_engine()
    query = _read_query(args.query)
    try:
        report = engine.consistency_check(query, args.parallelism)
    except ColStoreError as exc:
        logger.error("consistency_diverged", exc)
        return _emit(exc.to_dict(), code=1)
    logger.info("consistency_ok", report=report)
    return _emit(report)


def cmd_estimate(args: argparse.Namespace) -> int:
    engine = _load_engine()
    report = engine.encoding_estimate(args.segment, args.row_group, args.column)
    return _emit(report)


def cmd_inject(args: argparse.Namespace) -> int:
    logger = StructuredLogger(LOG_PATH)
    fault_path = os.path.join(SAMPLE_DIR, "fault_params.json")
    with open(fault_path, "r", encoding="utf-8") as handle:
        fault_params = json.load(handle)[args.kind]
    logger.info("fault_injection_started", kind=args.kind, params=fault_params)
    try:
        if args.kind == "zero_budget":
            zero = BudgetParams(
                max_parallelism=fault_params["max_parallelism"],
                memory_budget_bytes=fault_params["memory_budget_bytes"],
                per_segment_concurrency=fault_params["per_segment_concurrency"],
            )
            logger.info("zero_budget_apply", budget=zero.to_dict())
            ScanEngine(DATA_DIR, zero)  # constructor validates -> E_BUDGET
            raise AssertionError("zero budget was accepted")

        engine = _load_engine()
        segment_name = fault_params["segment"]
        source = engine.segment_path(segment_name)
        corrupted = os.path.join(DATA_DIR, f"corrupt_{args.kind}.colseg")
        if args.kind == "payload_truncation":
            cut = inject_payload_truncation(source, corrupted)
            logger.info("payload_truncated", output=corrupted, cut_at=cut)
            definition = engine.definitions[segment_name]
            load_segment(corrupted, definition)  # expected E_PAYLOAD
        elif args.kind == "encoding_mismatch":
            detail = inject_encoding_mismatch(source, corrupted)
            logger.info("encoding_flipped", output=corrupted, detail=detail)
            definition = engine.definitions[segment_name]
            load_segment(corrupted, definition)  # expected E_ENCODING
        else:  # pragma: no cover - argparse choices guard
            raise ValueError(args.kind)
    except ColStoreError as exc:
        # Completed block-level work remains verifiable via the engine history.
        verifiable = _verifiable_completed_tasks(locals().get("engine"))
        logger.error("fault_reported", exc, verifiable_completed_tasks=verifiable)
        payload = exc.to_dict()
        payload["verifiable_completed_tasks"] = verifiable
        return _emit(payload, code=2)
    return _emit({"status": "injection unexpectedly accepted", "kind": args.kind}, code=3)


def _verifiable_completed_tasks(engine: ScanEngine | None) -> List[Dict[str, Any]]:
    if engine is None:
        return []
    completed: List[Dict[str, Any]] = []
    for result in engine.generations.values():
        for task in result.tasks:
            if task.status == "done":
                completed.append(
                    {
                        "task_id": task.task_id,
                        "block": task.block.to_dict(),
                        "rows": task.rows,
                        "matched_rows": task.matched_rows,
                    }
                )
    return completed


def cmd_snapshot(args: argparse.Namespace) -> int:
    logger = StructuredLogger(LOG_PATH)
    engine = _load_engine()
    store = SnapshotStore(SNAPSHOT_DB)
    if args.snapshot_action == "save":
        query = _read_query(args.query)
        result = engine.scan(query)
        summaries = [engine.segments[name].file_summary() for name in engine.segments]
        for summary in summaries:
            summary["encoding_selection_version"] = engine.encoding_selection_version
        store.save_session(
            snapshot_id=args.id,
            result=result,
            budget=engine.budget,
            segment_summaries=summaries,
            encoding_version=engine.encoding_selection_version,
            current_state={"budget": engine.budget.to_dict()},
        )
        logger.info("snapshot_saved", snapshot_id=args.id)
        return _emit({"saved": args.id, "result_hash": result.result_hash})
    if args.snapshot_action == "open":
        current = [engine.segments[name].file_summary() for name in engine.segments]
        for summary in current:
            summary["encoding_selection_version"] = engine.encoding_selection_version
        diffs = store.compare_with_current(args.id, current, engine.encoding_selection_version)
        session = store.load_session(args.id)
        if diffs:
            new_id = f"reopened-{args.id}"
            if not any(s["snapshot_id"] == new_id for s in store.list_snapshots()):
                store.mark_superseded(args.id, new_id, diffs)
            logger.info("snapshot_replaced", old=args.id, diff_count=len(diffs))
        else:
            logger.info("snapshot_matched", snapshot_id=args.id)
        return _emit({"snapshot": session, "diffs": diffs, "replaced": bool(diffs)})
    if args.snapshot_action == "list":
        return _emit({"snapshots": store.list_snapshots()})
    return 4  # pragma: no cover


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Columnar store compression & scan system")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare").set_defaults(func=cmd_prepare)

    scan = sub.add_parser("scan")
    scan.add_argument("--query", default="query_and.json")
    scan.add_argument("--parallelism", type=int, default=4)
    scan.add_argument("--memory-budget", type=int, default=1 << 16)
    scan.add_argument("--segment-concurrency", type=int, default=2)
    scan.set_defaults(func=cmd_scan)

    check = sub.add_parser("check")
    check.add_argument("--query", default="query_eq.json")
    check.add_argument("--parallelism", type=int, default=4)
    check.set_defaults(func=cmd_check)

    estimate = sub.add_parser("estimate")
    estimate.add_argument("--segment", default="sales")
    estimate.add_argument("--row-group", type=int, default=0)
    estimate.add_argument("--column", default="region_id")
    estimate.set_defaults(func=cmd_estimate)

    inject = sub.add_parser("inject")
    inject.add_argument(
        "--kind",
        choices=["payload_truncation", "encoding_mismatch", "zero_budget"],
        required=True,
    )
    inject.set_defaults(func=cmd_inject)

    snapshot = sub.add_parser("snapshot")
    snap_sub = snapshot.add_subparsers(dest="snapshot_action", required=True)
    save = snap_sub.add_parser("save")
    save.add_argument("--id", required=True)
    save.add_argument("--query", default="query_eq.json")
    open_cmd = snap_sub.add_parser("open")
    open_cmd.add_argument("--id", required=True)
    snap_sub.add_parser("list")
    snapshot.set_defaults(func=cmd_snapshot)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ColStoreError as exc:
        StructuredLogger(LOG_PATH).error("cli_business_error", exc)
        return _emit(exc.to_dict(), code=1)


if __name__ == "__main__":
    sys.exit(main())
