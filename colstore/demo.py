"""End-to-end demonstration driver.

Runs the three acceptance scenarios in order and writes:

* ``data/demo.log.jsonl`` - structured per-event log;
* ``data/demo_summary.json`` - machine-readable verification summary;
* ``data/demo_summary.md`` - human-readable verification report.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .engine import ScanEngine
from .errors import ColStoreError
from .model import BudgetParams, Encoding, SegmentDefinition
from .prepare_samples import DATA_DIR, SAMPLE_DIR
from .snapshot import SnapshotStore
from .structured_log import StructuredLogger

SUMMARY_JSON = os.path.join(DATA_DIR, "demo_summary.json")
SUMMARY_MD = os.path.join(DATA_DIR, "demo_summary.md")


def _load_definitions(engine: ScanEngine) -> None:
    for filename in sorted(os.listdir(SAMPLE_DIR)):
        if filename.startswith("segment_") and filename.endswith(".json"):
            with open(os.path.join(SAMPLE_DIR, filename), "r", encoding="utf-8") as handle:
                engine.register_definition(SegmentDefinition.from_dict(json.load(handle)))


def _read_query(name: str) -> Dict[str, Any]:
    with open(os.path.join(SAMPLE_DIR, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_demo() -> Dict[str, Any]:
    os.makedirs(DATA_DIR, exist_ok=True)
    for stale in ("demo.log.jsonl", "demo_summary.json", "demo_summary.md", "sessions.db"):
        path = os.path.join(DATA_DIR, stale)
        if os.path.exists(path):
            os.remove(path)
    logger = StructuredLogger(os.path.join(DATA_DIR, "demo.log.jsonl"))
    summary: Dict[str, Any] = {"scenarios": {}, "checks": [], "verdict": "PENDING"}

    def check(name: str, ok: bool, detail: Any) -> None:
        summary["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        logger.info("verification", check=name, ok=bool(ok), detail=detail)

    initial_budget = BudgetParams(
        max_parallelism=4, memory_budget_bytes=1 << 16, per_segment_concurrency=2
    )
    engine = ScanEngine(DATA_DIR, initial_budget)
    _load_definitions(engine)
    engine.build_segments()
    query = _read_query("query_and.json")
    variant_query = _read_query("query_and_variant.json")

    # ------------------------------------------------------------------
    # Scenario 1: first scan (column pruning + push-down + scheduling)
    # ------------------------------------------------------------------
    logger.info("scenario_1_first_scan", budget=initial_budget.to_dict())
    first = engine.scan(query)
    variant = engine.scan(variant_query)
    check(
        "canonical_query_id_stable_under_whitespace_and_parentheses",
        first.query_id == variant.query_id,
        {"query_id": first.query_id},
    )
    check(
        "column_pruning_excludes_unreferenced_columns",
        set(first.plan_summary["pruned_columns"]) == {"sku_id", "warehouse_id", "qty"},
        {"pruned": first.plan_summary["pruned_columns"]},
    )
    peak_ok = first.stats.peak_memory_bytes <= initial_budget.memory_budget_bytes
    check(
        "scheduler_never_exceeds_memory_budget",
        peak_ok,
        {
            "peak_memory_bytes": first.stats.peak_memory_bytes,
            "budget_bytes": initial_budget.memory_budget_bytes,
        },
    )
    check(
        "all_tasks_record_lifecycle_timestamps",
        all(t.start_ms > 0 and t.end_ms >= t.start_ms for t in first.tasks),
        {"task_count": len(first.tasks)},
    )
    summary["scenarios"]["first_scan"] = {
        "result_id": first.result_id,
        "rows_matched": first.stats.rows_matched,
        "result_hash": first.result_hash,
        "task_timeline": [t.to_dict() for t in first.tasks],
    }

    # ------------------------------------------------------------------
    # Scenario 2: budget change -> version bump, stale guard, reschedule
    # ------------------------------------------------------------------
    old_id = first.result_id
    changed_budget = BudgetParams(
        max_parallelism=2, memory_budget_bytes=4096, per_segment_concurrency=1
    )
    logger.info("scenario_2_budget_change", budget=changed_budget.to_dict())
    replacement = engine.update_budget(changed_budget)
    check(
        "budget_version_monotonic_increase",
        engine.budget_version == 2 and replacement.budget_version == 2,
        {"budget_version": engine.budget_version},
    )
    check(
        "previous_generation_marked_stale",
        engine.generations[old_id].stats.stale,
        {"result_id": old_id},
    )
    check(
        "replacement_lists_affected_blocks",
        len(replacement.affected_blocks) == first.stats.tasks_total,
        {"affected": len(replacement.affected_blocks)},
    )
    second = engine.scan(query)
    check(
        "byte_identical_results_after_reschedule",
        second.result_hash == first.result_hash and second.rows == first.rows,
        {"first_hash": first.result_hash, "second_hash": second.result_hash},
    )
    check(
        "stale_generation_blocked_from_summary",
        engine.current_summary()["current"]["result_id"] == second.result_id,
        {"current_result_id": second.result_id},
    )
    summary["scenarios"]["budget_change"] = {
        "replacement": replacement.to_dict(),
        "new_result_id": second.result_id,
        "rows_matched": second.stats.rows_matched,
    }

    # ------------------------------------------------------------------
    # Scenario 3: snapshot save, reopen, whole-replacement verification
    # ------------------------------------------------------------------
    logger.info("scenario_3_snapshot")
    store = SnapshotStore(os.path.join(DATA_DIR, "sessions.db"))
    snapshot_id = "demo-session-1"
    summaries = [engine.segments[name].file_summary() for name in engine.segments]
    for item in summaries:
        item["encoding_selection_version"] = engine.encoding_selection_version
    store.save_session(
        snapshot_id=snapshot_id,
        result=second,
        budget=engine.budget,
        segment_summaries=summaries,
        encoding_version=engine.encoding_selection_version,
        current_state={"budget": engine.budget.to_dict()},
    )
    same_diffs = store.compare_with_current(
        snapshot_id, summaries, engine.encoding_selection_version
    )
    check("snapshot_matches_disk_when_unchanged", same_diffs == [], {"diffs": same_diffs})

    # Simulate an on-disk encoding selection change: rebuild with forced RLE,
    # then reopen and assert whole replacement + read-only old snapshot.
    engine.rebuild_with_encoding(Encoding.RLE)
    changed_summaries = [engine.segments[name].file_summary() for name in engine.segments]
    for item in changed_summaries:
        item["encoding_selection_version"] = engine.encoding_selection_version
    diffs = store.compare_with_current(
        snapshot_id, changed_summaries, engine.encoding_selection_version
    )
    check(
        "snapshot_detects_encoding_version_drift",
        any(diff["kind"] == "encoding_version_changed" for diff in diffs),
        {"diff_kinds": [diff["kind"] for diff in diffs]},
    )
    new_id = "demo-session-2"
    store.mark_superseded(snapshot_id, new_id, diffs)
    reopened = store.load_session(snapshot_id)
    check(
        "superseded_snapshot_frozen_read_only_with_diff_list",
        reopened["read_only"] and len(reopened["replacement_diffs"]) == len(diffs),
        {"diff_count": len(diffs)},
    )
    restored_result = ScanResult_safe(reopened)
    check(
        "restored_result_hash_matches_original_snapshot",
        restored_result["result_hash"] == second.result_hash,
        {"restored_hash": restored_result["result_hash"]},
    )
    store.close()
    summary["scenarios"]["snapshot_restore"] = {
        "snapshot_id": snapshot_id,
        "diff_count": len(diffs),
        "diffs": diffs,
    }

    summary["verdict"] = "PASS" if all(item["ok"] for item in summary["checks"]) else "FAIL"
    with open(SUMMARY_JSON, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=str)
    _write_markdown(summary)
    logger.info("demo_finished", verdict=summary["verdict"], checks=len(summary["checks"]))
    return summary


def ScanResult_safe(session: Dict[str, Any]) -> Dict[str, Any]:
    return session["result"]


def _write_markdown(summary: Dict[str, Any]) -> None:
    lines = [
        "# Columnar Store Demo Verification Summary",
        "",
        f"Verdict: **{summary['verdict']}**",
        "",
        "## Checks",
    ]
    for item in summary["checks"]:
        mark = "PASS" if item["ok"] else "FAIL"
        lines.append(f"- [{mark}] {item['name']}: {json.dumps(item['detail'], ensure_ascii=False)}")
    lines.extend(["", "## Scenarios"])
    for name, detail in summary["scenarios"].items():
        lines.append(f"### {name}")
        lines.append("```json")
        lines.append(json.dumps(detail, indent=2, ensure_ascii=False, default=str)[:4000])
        lines.append("```")
    with open(SUMMARY_MD, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    report = run_demo()
    print(json.dumps({"verdict": report["verdict"], "checks": len(report["checks"])}, indent=2))
