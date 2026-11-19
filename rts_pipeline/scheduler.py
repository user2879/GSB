"""Budget-aware parallel block synthesis and deterministic merging."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .canonical import hash_value
from .chunker import load_chunk
from .constants import SCHEDULER_VERSION, SCHEMA_VERSION
from .engine import synthesize_range
from .errors import PipelineError, require
from .rules import rules_hash, validate_rules


def _synthesize_chunk(
    root: Path,
    manifest_entry: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    chunk = load_chunk(root, manifest_entry)
    expected_rules_hash = rules_hash(rules)
    if chunk["rules_hash"] != expected_rules_hash:
        raise PipelineError(
            "E_CHUNK_BOUNDARY",
            "chunk rules hash differs from active rules",
            chunk_id=chunk["chunk_id"],
            expected=expected_rules_hash,
            actual=chunk["rules_hash"],
        )
    result = synthesize_range(
        events=chunk["events"],
        rules=rules,
        seed=chunk["game_seed"],
        start_tick=chunk["start_tick"],
        end_tick=chunk["end_tick"],
        entry_snapshot=chunk["entry_snapshot"],
    )
    if result["entry_snapshot_hash"] != manifest_entry["entry_snapshot_hash"]:
        raise PipelineError(
            "E_CHUNK_BOUNDARY",
            "computed entry snapshot hash differs from manifest",
            chunk_id=chunk["chunk_id"],
            expected=manifest_entry["entry_snapshot_hash"],
            actual=result["entry_snapshot_hash"],
        )
    if result["exit_snapshot_hash"] != manifest_entry["exit_snapshot_hash"]:
        raise PipelineError(
            "E_CHUNK_BOUNDARY",
            "computed exit snapshot differs from declared chunk boundary",
            chunk_id=chunk["chunk_id"],
            expected=manifest_entry["exit_snapshot_hash"],
            actual=result["exit_snapshot_hash"],
        )
    if hash_value(chunk["exit_snapshot"]) != manifest_entry["exit_snapshot_hash"]:
        raise PipelineError(
            "E_CHUNK_BOUNDARY",
            "declared exit snapshot is inconsistent",
            chunk_id=chunk["chunk_id"],
        )
    return {
        "chunk_id": chunk["chunk_id"],
        "start_tick": chunk["start_tick"],
        "end_tick": chunk["end_tick"],
        "entry_snapshot": chunk["entry_snapshot"],
        "exit_snapshot": chunk["exit_snapshot"],
        "entry_snapshot_hash": result["entry_snapshot_hash"],
        "computed_exit_snapshot": result["exit_snapshot"],
        "computed_exit_snapshot_hash": result["exit_snapshot_hash"],
        "declared_exit_snapshot_hash": manifest_entry["exit_snapshot_hash"],
        "tick_records": result["tick_records"],
        "event_records": result["event_records"],
        "violations": result["violations"],
        "rng_records": result["rng_records"],
        "event_count": result["event_count"],
        "result_codes": result["result_codes"],
    }


def _budget_plan(
    chunk_count: int,
    requested_threads: int,
    memory_budget_bytes: int,
    max_block_bytes: int,
    mode: str,
) -> dict[str, Any]:
    degradations: list[dict[str, Any]] = []
    requested_parallelism = (
        1 if mode == "serial" else max(0, min(requested_threads, chunk_count))
    )
    effective = requested_parallelism
    denial_reason: str | None = None
    if mode == "serial" and requested_threads > 1:
        degradations.append(
            {
                "priority": 1,
                "reason": "serial_mode",
                "before_parallelism": min(requested_threads, chunk_count),
                "after_parallelism": 1,
            }
        )
    if effective > chunk_count:
        before = effective
        effective = chunk_count
        degradations.append(
            {
                "priority": 2,
                "reason": "chunk_count_limit",
                "before_parallelism": before,
                "after_parallelism": effective,
            }
        )
    if requested_threads <= 0:
        before = effective
        effective = 0
        denial_reason = "thread_limit_exhausted"
        degradations.append(
            {
                "priority": 3,
                "reason": denial_reason,
                "before_parallelism": before,
                "after_parallelism": 0,
                "requested_threads": requested_threads,
            }
        )
    elif memory_budget_bytes < max_block_bytes:
        before = effective
        effective = 0
        denial_reason = "memory_budget_below_single_block"
        degradations.append(
            {
                "priority": 4,
                "reason": denial_reason,
                "before_parallelism": before,
                "after_parallelism": 0,
                "memory_budget_bytes": memory_budget_bytes,
                "required_block_bytes": max_block_bytes,
            }
        )
    elif max_block_bytes > 0:
        memory_workers = memory_budget_bytes // max_block_bytes
        if memory_workers < effective:
            before = effective
            effective = max(0, memory_workers)
            degradations.append(
                {
                    "priority": 4,
                    "reason": "memory_budget_limits_parallelism",
                    "before_parallelism": before,
                    "after_parallelism": effective,
                    "memory_budget_bytes": memory_budget_bytes,
                    "memory_capacity_workers": memory_workers,
                    "required_block_bytes": max_block_bytes,
                }
            )
    return {
        "requested_parallelism": requested_parallelism,
        "effective_parallelism": effective,
        "degradations": degradations,
        "denial_reason": denial_reason,
    }


def run_scheduler(
    root: Path,
    manifest: dict[str, Any],
    rules: dict[str, Any],
    requested_threads: int,
    memory_budget_bytes: int,
    *,
    mode: str = "parallel",
) -> dict[str, Any]:
    require(mode in ("parallel", "serial"), "E_SCHEMA", "scheduler mode is invalid", mode=mode)
    normalized_rules = validate_rules(rules)
    entries = sorted(manifest["chunks"], key=lambda item: (item["start_tick"], item["chunk_id"]))
    chunk_count = len(entries)
    max_block_bytes = max((entry["estimated_memory_bytes"] for entry in entries), default=0)
    plan = _budget_plan(
        chunk_count=chunk_count,
        requested_threads=requested_threads,
        memory_budget_bytes=memory_budget_bytes,
        max_block_bytes=max_block_bytes,
        mode=mode,
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scheduler_version": SCHEDULER_VERSION,
        "mode": mode,
        "requested_threads": requested_threads,
        "requested_parallelism": plan["requested_parallelism"],
        "effective_parallelism": plan["effective_parallelism"],
        "memory_budget_bytes": memory_budget_bytes,
        "max_block_estimated_bytes": max_block_bytes,
        "chunk_count": chunk_count,
        "accepted_chunks": 0,
        "denied_chunks": 0,
        "completed_chunks": 0,
        "degradations": plan["degradations"],
        "denial_reason": plan["denial_reason"],
        "denied_blocks": [],
        "task_failures": [],
        "merge_order_basis": "start_tick_then_chunk_id",
        "result_codes": [],
        "status": "running",
    }
    if plan["effective_parallelism"] < 1:
        report["status"] = "denied"
        report["result_codes"] = ["E_BUDGET_DENIED"]
        report["denied_chunks"] = chunk_count
        report["denied_blocks"] = [
            {
                "chunk_id": entry["chunk_id"],
                "start_tick": entry["start_tick"],
                "end_tick": entry["end_tick"],
                "code": "E_BUDGET_DENIED",
                "reason": plan["denial_reason"],
                "required_memory_bytes": entry["estimated_memory_bytes"],
                "available_memory_bytes": memory_budget_bytes,
            }
            for entry in entries
        ]
        return {
            "status": "denied",
            "error_code": "E_BUDGET_DENIED",
            "results": [],
            "report": report,
        }

    report["accepted_chunks"] = chunk_count
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if plan["effective_parallelism"] == 1:
        for entry in entries:
            try:
                results.append(_synthesize_chunk(Path(root), entry, normalized_rules))
            except PipelineError as exc:
                failures.append(
                    {
                        "chunk_id": entry["chunk_id"],
                        "start_tick": entry["start_tick"],
                        "error": exc.to_dict(),
                    }
                )
                break
    else:
        with ThreadPoolExecutor(
            max_workers=plan["effective_parallelism"],
            thread_name_prefix="rts-block",
        ) as executor:
            futures = {
                executor.submit(_synthesize_chunk, Path(root), entry, normalized_rules): entry
                for entry in entries
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    results.append(future.result())
                except PipelineError as exc:
                    failures.append(
                        {
                            "chunk_id": entry["chunk_id"],
                            "start_tick": entry["start_tick"],
                            "error": exc.to_dict(),
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "chunk_id": entry["chunk_id"],
                            "start_tick": entry["start_tick"],
                            "error": {
                                "code": "E_INTERNAL",
                                "message": str(exc),
                                "details": {},
                            },
                        }
                    )
    results.sort(key=lambda item: (item["start_tick"], item["chunk_id"]))
    failures.sort(key=lambda item: (item["start_tick"], item["chunk_id"]))
    report["completed_chunks"] = len(results)
    report["task_failures"] = failures
    if failures:
        report["status"] = "failed"
        report["result_codes"] = [failures[0]["error"]["code"]]
        return {
            "status": "failed",
            "error_code": failures[0]["error"]["code"],
            "error": failures[0],
            "results": results,
            "report": report,
        }
    report["status"] = "ok"
    overflow = any(result["result_codes"] for result in results)
    report["result_codes"] = ["E_RESOURCE_OVERFLOW"] if overflow else []
    return {
        "status": "ok",
        "error_code": None,
        "results": results,
        "report": report,
    }

