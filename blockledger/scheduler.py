from __future__ import annotations

import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .blocks import ContentStore, MaterializedBlocks
from .clockutil import monotonic_ms
from .sampling import is_selected
from .verification import verify_block_with_subprocess


BlockKey = tuple[str, int]


def make_tasks(
    replay_plans: list[dict[str, Any]],
    budget: dict[str, int],
    *,
    attempt: int = 1,
    unknown_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    max_blocks = int(budget["max_blocks_per_rollback"])
    global_concurrency = int(budget["global_rollback_concurrency"])
    per_volume = int(budget["per_volume_rollback_concurrency"])

    if unknown_rows is not None:
        grouped: dict[str, list[dict[str, Any]]] = {}
        metadata: dict[str, dict[str, Any]] = {}
        for row in unknown_rows:
            rollback_id = row["rollback_id"]
            grouped.setdefault(rollback_id, []).append(
                {
                    "block_index": int(row["block_index"]),
                    "expected_digest": row["expected_digest"],
                    "baseline_version": int(row["baseline_version"]),
                    "replay_block_version": int(row["block_version"]),
                }
            )
            metadata[rollback_id] = {
                "event_seq": 0,
                "volume_id": row["volume_id"],
                "snapshot_id": "recovery",
            }
        for rollback_id in sorted(grouped):
            blocks = sorted(grouped[rollback_id], key=lambda item: item["block_index"])
            for chunk_index, start in enumerate(range(0, len(blocks), max_blocks)):
                chunk = blocks[start : start + max_blocks]
                tasks.append(
                    {
                        "batch_id": f"{rollback_id}:rerun-a{attempt:02d}-b{chunk_index:03d}",
                        "rollback_id": rollback_id,
                        "event_seq": metadata[rollback_id]["event_seq"],
                        "volume_id": metadata[rollback_id]["volume_id"],
                        "snapshot_id": metadata[rollback_id]["snapshot_id"],
                        "batch_index": chunk_index,
                        "blocks": chunk,
                        "max_blocks": max_blocks,
                        "global_concurrency": global_concurrency,
                        "per_volume_concurrency": per_volume,
                        "wait_started_ms": monotonic_ms(),
                        "wait_ended_ms": None,
                        "rerun": True,
                    }
                )
        return tasks

    for plan in sorted(replay_plans, key=lambda item: (item["event_seq"], item["volume_id"], item["rollback_id"])):
        blocks = sorted(plan["blocks"], key=lambda item: item["block_index"])
        plan_limit = max_blocks
        if plan.get("max_blocks"):
            plan_limit = min(max_blocks, int(plan["max_blocks"]))
        for batch_index, start in enumerate(range(0, len(blocks), plan_limit)):
            chunk = blocks[start : start + plan_limit]
            tasks.append(
                {
                    "batch_id": f"{plan['rollback_id']}:a{attempt:02d}-b{batch_index:03d}",
                    "rollback_id": plan["rollback_id"],
                    "event_seq": plan["event_seq"],
                    "volume_id": plan["volume_id"],
                    "snapshot_id": plan["snapshot_id"],
                    "batch_index": batch_index,
                    "blocks": chunk,
                        "max_blocks": plan_limit,
                    "global_concurrency": global_concurrency,
                    "per_volume_concurrency": per_volume,
                    "wait_started_ms": monotonic_ms(),
                    "wait_ended_ms": None,
                    "rerun": False,
                }
            )
    tasks.sort(key=lambda task: (task["event_seq"], task["volume_id"], task["rollback_id"], task["batch_index"]))
    return tasks


def execute_task(
    task: dict[str, Any],
    *,
    ledger: Any,
    content_store: ContentStore,
    work_root: str | Path,
    sample_permille: int,
    timeout_ms: int,
    fault_kill_blocks: set[BlockKey] | None = None,
    corrupt_blocks: set[BlockKey] | None = None,
    crash_after_lease_batch: str | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    fault_kill_blocks = fault_kill_blocks or set()
    corrupt_blocks = corrupt_blocks or set()
    ledger.start_batch_lease(task, task["_global_active"], task["_volume_active"])
    if crash_after_lease_batch == task["batch_id"]:
        # Real immediate process death after the lease transaction is durable.
        os.kill(os.getpid(), signal.SIGKILL)

    work = MaterializedBlocks(Path(work_root) / task["rollback_id"] / task["batch_id"])
    outcomes: list[dict[str, Any]] = []
    for block in task["blocks"]:
        index = int(block["block_index"])
        volume_id = task["volume_id"]
        expected = block["expected_digest"]
        selected = is_selected(task["rollback_id"], volume_id, index, sample_permille)
        base_outcome = {
            "volume_id": volume_id,
            "block_index": index,
            "block_version": int(block["replay_block_version"]),
            "baseline_version": int(block["baseline_version"]),
            "expected_digest": expected,
            "restored_digest": expected,
        }
        if not selected:
            outcomes.append(
                {
                    **base_outcome,
                    "sample_selected": False,
                    "verification_status": "unchecked",
                    "reason": "deterministic_sample_not_selected",
                    "restored_digest": expected,
                }
            )
            continue

        content = content_store.get(expected)
        if content is None:
            outcomes.append(
                {
                    **base_outcome,
                    "sample_selected": True,
                    "verification_status": "unknown",
                    "reason": "content_store_missing_digest",
                    "restored_digest": None,
                }
            )
            continue
        block_file = work.write(volume_id, index, content)
        verification = verify_block_with_subprocess(
            block_file=block_file,
            volume_id=volume_id,
            block_index=index,
            baseline_version=int(block["baseline_version"]),
            expected_digest=expected,
            timeout_ms=timeout_ms,
            force_kill=(volume_id, index) in fault_kill_blocks,
            corrupt_in_memory=(volume_id, index) in corrupt_blocks,
            python_executable=python_executable,
        )
        outcomes.append(
            {
                **base_outcome,
                "sample_selected": True,
                "verification_status": verification["verification_status"],
                "restored_digest": verification.get("restored_digest"),
                "reason": verification.get("reason"),
                "worker_pid": verification.get("worker_pid"),
                "worker_returncode": verification.get("worker_returncode"),
                "force_killed": verification.get("force_killed", False),
            }
        )

    if any(item["verification_status"] == "corrupt" for item in outcomes):
        status = "corrupt"
    elif any(item["verification_status"] == "unknown" for item in outcomes):
        status = "unknown"
    else:
        status = "completed"
    committed = ledger.complete_batch(task["batch_id"], outcomes, status=status)
    return {"task": task, "outcomes": outcomes, "commit": committed}


def run_quota_schedule(
    tasks: list[dict[str, Any]],
    *,
    ledger: Any,
    content_store: ContentStore,
    work_root: str | Path,
    sample_permille: int,
    timeout_ms: int,
    fault_kill_blocks: set[BlockKey] | None = None,
    corrupt_blocks: set[BlockKey] | None = None,
    crash_after_lease_batch: str | None = None,
    python_executable: str | None = None,
    before_task: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if not tasks:
        return []
    global_limit = int(tasks[0]["global_concurrency"])
    volume_limit = int(tasks[0]["per_volume_concurrency"])
    max_workers = max(1, min(global_limit, len(tasks)))
    pending = sorted(tasks, key=lambda task: (task["event_seq"], task["volume_id"], task["batch_index"]))
    condition = threading.Condition()
    active_global = 0
    active_by_volume: dict[str, int] = {}
    in_flight: dict[Any, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def schedule_locked(executor: ThreadPoolExecutor) -> None:
        nonlocal active_global
        remaining: list[dict[str, Any]] = []
        for task in pending:
            volume_id = task["volume_id"]
            volume_active = active_by_volume.get(volume_id, 0)
            if active_global < global_limit and volume_active < volume_limit:
                active_global += 1
                active_by_volume[volume_id] = volume_active + 1
                task["wait_ended_ms"] = monotonic_ms()
                task["_global_active"] = active_global
                task["_volume_active"] = volume_active + 1
                if before_task is not None:
                    before_task(task)

                future = executor.submit(
                    execute_task,
                    task,
                    ledger=ledger,
                    content_store=content_store,
                    work_root=work_root,
                    sample_permille=sample_permille,
                    timeout_ms=timeout_ms,
                    fault_kill_blocks=fault_kill_blocks,
                    corrupt_blocks=corrupt_blocks,
                    crash_after_lease_batch=crash_after_lease_batch,
                    python_executable=python_executable,
                )
                in_flight[future] = task
            else:
                remaining.append(task)
        pending[:] = remaining

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rollback-quota") as executor:
        with condition:
            schedule_locked(executor)
            while in_flight or pending:
                condition.wait(timeout=0.02)
                finished = [future for future in list(in_flight) if future.done()]
                for future in finished:
                    task = in_flight.pop(future)
                    active_global -= 1
                    volume_id = task["volume_id"]
                    active_by_volume[volume_id] = max(0, active_by_volume.get(volume_id, 1) - 1)
                    try:
                        results.append(future.result())
                    except BaseException as exc:  # preserve and re-raise after pool shutdown
                        errors.append(exc)
                    condition.notify_all()
                if not in_flight and pending:
                    schedule_locked(executor)
                elif pending:
                    schedule_locked(executor)

    if errors:
        raise errors[0]
    results.sort(key=lambda item: item["task"]["batch_id"])
    return results
