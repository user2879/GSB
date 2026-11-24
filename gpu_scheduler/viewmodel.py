from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import canonical_bytes, digest_obj
from .ledger import Ledger
from .quant import build_watermarks
from .scheduler import SchedulerEngine
from .spec import Spec


def build_engine_view(engine: SchedulerEngine) -> dict[str, Any]:
    report = engine.state_report()
    preemptions = [
        dict(row)
        for row in engine.ledger.conn.execute(
            "SELECT round_index,high_task_id,victim_task_id,victim_set_json,selection_rule,reason "
            "FROM preemption_rows ORDER BY id"
        )
    ]
    suspensions = [
        dict(row)
        for row in engine.ledger.conn.execute(
            "SELECT task_id,round_index,pool_id,state,completed_shards_json,cursor_json,"
            "renew_count,recovery_path FROM suspension_rows ORDER BY task_id"
        )
    ]
    preempted_tasks = sorted({row["victim_task_id"] for row in preemptions})
    return {
        "format_version": 1,
        "round_index": engine.head_round,
        "watermarks": engine._build_watermarks(),
        "counts": report["counts"],
        "running": report["running"],
        "queued": report["queued"],
        "unknown": report["unknown"],
        "active": report["active"],
        "preempted_tasks": preempted_tasks,
        "preemptions": preemptions,
        "suspensions": suspensions,
    }


def write_ui_cache(engine: SchedulerEngine, path: Path | str | None = None) -> dict[str, Any]:
    view = build_engine_view(engine)
    target = Path(path or engine.workdir / "ui-cache.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_bytes(view) + b"\n")
    return view


def build_ledger_view(spec: Spec, ledger: Ledger) -> dict[str, Any]:
    ledger.verify_digests()
    head = ledger.head_round
    occupancy_rows = ledger.conn.execute(
        "SELECT * FROM occupancy_rows WHERE round_index=? ORDER BY scope DESC, pool_id",
        (head,),
    ).fetchall()
    gpu_used: dict[str, int] = {}
    mem_used: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for row in occupancy_rows:
        if row["scope"] == "pool":
            gpu_used[row["pool_id"]] = row["used_gpu"]
            mem_used[row["pool_id"]] = row["used_memory_mib"]
    # Active task count is authoritatively derived from nonterminal dispatch events.
    latest_phase: dict[str, str] = {}
    active_tasks_by_pool: dict[str, set[str]] = {}
    for event in ledger.conn.execute("SELECT shard_id,task_id,phase,payload_json FROM shard_events ORDER BY id"):
        latest_phase[event["shard_id"]] = event["phase"]
        if event["phase"] == "dispatched":
            task = spec.task_by_id.get(event["task_id"])
            if task:
                active_tasks_by_pool.setdefault(task.pool, set()).add(event["task_id"])
        if event["phase"] in {"completed", "failed"}:
            task = spec.task_by_id.get(event["task_id"])
            if task:
                active_tasks_by_pool.get(task.pool, set()).discard(event["task_id"])
    for shard_id, row in ledger.latest_receipts().items():
        latest_phase[shard_id] = "completed" if row["outcome_status"] == "success" else "failed"
    task_counts = {pool_id: len(tasks) for pool_id, tasks in active_tasks_by_pool.items()}
    watermarks = build_watermarks(spec, gpu_used, mem_used, task_counts)

    running: list[str] = []
    queued: list[str] = []
    unknown: list[str] = []
    counts: dict[str, int] = {}
    for shard_id in sorted(latest_phase):
        phase = latest_phase[shard_id]
        counts[phase] = counts.get(phase, 0) + 1
        if phase in {"dispatched", "running"}:
            running.append(shard_id)
        elif phase == "queued":
            queued.append(shard_id)
        elif phase == "unknown":
            unknown.append(shard_id)

    preemptions = [
        dict(row)
        for row in ledger.conn.execute(
            "SELECT round_index,high_task_id,victim_task_id,victim_set_json,selection_rule,reason "
            "FROM preemption_rows ORDER BY id"
        )
    ]
    suspensions = [
        dict(row)
        for row in ledger.conn.execute(
            "SELECT task_id,round_index,pool_id,state,completed_shards_json,cursor_json,"
            "renew_count,recovery_path FROM suspension_rows ORDER BY task_id"
        )
    ]
    preempted_tasks = sorted({row["victim_task_id"] for row in preemptions})
    return {
        "format_version": 1,
        "round_index": head,
        "watermarks": watermarks,
        "counts": dict(sorted(counts.items())),
        "running": running,
        "queued": queued,
        "unknown": unknown,
        "active": running,
        "preempted_tasks": preempted_tasks,
        "preemptions": preemptions,
        "suspensions": suspensions,
    }


def compare_cache_file(spec: Spec, workdir: Path | str, cache_path: Path | str | None = None) -> dict[str, Any]:
    workdir = Path(workdir)
    cache_path = Path(cache_path or workdir / "ui-cache.json")
    ledger = Ledger(workdir / "ledger.sqlite3")
    try:
        authoritative = build_ledger_view(spec, ledger)
    finally:
        ledger.close()
    auth_bytes = canonical_bytes(authoritative)
    try:
        cached_raw = cache_path.read_bytes()
        cached_obj = json.loads(cached_raw.decode("utf-8"))
        cache_bytes = canonical_bytes(cached_obj)
        cache_error = None
    except FileNotFoundError:
        cached_obj = None
        cache_bytes = b""
        cache_error = "cache_file_missing"
    except Exception as exc:
        cached_obj = None
        cache_bytes = b""
        cache_error = f"cache_parse_error:{exc}"
    equal = cache_bytes == auth_bytes
    first_difference = None
    if not equal and cached_obj is not None:
        first_difference = _first_diff(cached_obj, authoritative)
    return {
        "byte_equal": equal,
        "cache_error": cache_error,
        "cache_sha256": digest_obj(cached_obj) if cached_obj is not None else None,
        "authoritative_sha256": digest_obj(authoritative),
        "cache_bytes": len(cache_bytes),
        "authoritative_bytes": len(auth_bytes),
        "first_difference": first_difference,
        "authoritative_view": authoritative,
    }


def _first_diff(cached: Any, authoritative: Any, path: str = "$") -> dict[str, Any] | None:
    if type(cached) is not type(authoritative):
        return {"path": path, "expected_type": type(authoritative).__name__, "cached_type": type(cached).__name__}
    if isinstance(authoritative, dict):
        for key in sorted(set(authoritative) | set(cached or {})):
            next_path = f"{path}.{key}"
            if key not in cached:
                return {"path": next_path, "issue": "missing_in_cache"}
            if key not in authoritative:
                return {"path": next_path, "issue": "unexpected_in_cache"}
            diff = _first_diff(cached[key], authoritative[key], next_path)
            if diff:
                return diff
    elif isinstance(authoritative, list):
        if len(cached) != len(authoritative):
            return {"path": path, "issue": "length_mismatch", "cached_length": len(cached), "authoritative_length": len(authoritative)}
        for index, (left, right) in enumerate(zip(cached, authoritative)):
            diff = _first_diff(left, right, f"{path}[{index}]")
            if diff:
                return diff
    elif cached != authoritative:
        return {"path": path, "cached": cached, "authoritative": authoritative}
    return None
