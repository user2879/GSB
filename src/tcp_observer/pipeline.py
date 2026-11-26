from __future__ import annotations

from pathlib import Path
from typing import Any

from .engine import build_observation
from .ledger import INTEGER_FORMAT_VERSION, Ledger
from .scheduler import InFlightScheduler, ScriptedMonotonicClock


def connection_id(spec: Any) -> str:
    return f"tcp-{spec.session_id}-127.0.0.1"


def schedule_payload_for_batch(batch_index: int, batch: dict[str, Any], spec: Any) -> dict[str, Any]:
    conn = connection_id(spec)
    clock_start = 0 if batch_index == 0 else 2 * batch_index - 1
    clock = ScriptedMonotonicClock(clock_start)
    scheduler = InFlightScheduler(
        global_limit=spec.global_inflight_limit,
        connection_quota=spec.connection_quota,
        clock=clock,
    )
    synthetic_predecessor: Any = None
    if batch_index > 0:
        synthetic_predecessor = scheduler.acquire(f"{batch['batch_id']}-preceding-commit", conn)
    ticket = scheduler.try_acquire(batch["batch_id"], conn)
    if batch_index > 0:
        assert ticket is not None and not ticket.admitted
        clock.advance(1)
        assert synthetic_predecessor is not None
        scheduler.release(synthetic_predecessor)
    assert ticket is not None and ticket.admitted
    scheduler.release(ticket)
    payload = {
        "integer_format_version": INTEGER_FORMAT_VERSION,
        "batch_index": batch_index,
        "batch_id": batch["batch_id"],
        "connection_id": conn,
        "global_inflight_limit": spec.global_inflight_limit,
        "connection_quota": spec.connection_quota,
        "monotonic_clock": "ScriptedMonotonicClock:non_decreasing_integer_milliseconds",
        "ticket": {
            "sequence": ticket.sequence,
            "queued": ticket.queued,
            "wait_start_ms": ticket.wait_start_ms,
            "wait_end_ms": ticket.wait_end_ms,
            "wait_ms": ticket.wait_ms,
        },
        "trajectory": scheduler.trajectory(),
        "batch_content_digest": batch["content_digest"],
    }
    return payload


def commit_observation(
    spec: Any,
    result: dict[str, Any],
    db_path: str | Path,
    *,
    crash_before_batch: int | None = None,
    reset: bool = False,
) -> Ledger:
    db_path = Path(db_path)
    if reset and db_path.exists():
        db_path.unlink()
    for suffix in ("-journal", "-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if reset and sidecar.exists():
            sidecar.unlink()
    ledger = Ledger(db_path)
    ledger.set_meta("integer_format_version", INTEGER_FORMAT_VERSION)
    ledger.set_meta("session_id", spec.session_id)
    ledger.set_meta("script_version", spec.script_version)
    ledger.set_meta("scale_version", spec.scale_version)
    ledger.set_meta("script_digest", spec.script_digest)
    ledger.set_meta("scale_caliber_digest", result["summary"]["scale_caliber_digest"])
    ledger.set_meta("summary_content_digest", result["summary"]["content_digest"])
    items_by_batch: dict[int, list[dict[str, Any]]] = {}
    for item in result["items"]:
        items_by_batch.setdefault(int(item["batch_index"]), []).append(item)

    for batch in result["batches"]:
        index = int(batch["batch_index"])
        payload = schedule_payload_for_batch(index, batch, spec)
        cursor = {
            "integer_format_version": INTEGER_FORMAT_VERSION,
            "last_committed_batch_index": index,
            "last_committed_event_seq": int(batch["end_seq"]),
            "next_event_seq": int(batch["end_seq"]) + 1,
            "summary_content_digest": result["summary"]["content_digest"],
        }
        ledger.commit_batch(
            batch,
            items_by_batch[index],
            payload,
            cursor,
            crash_before_commit=(crash_before_batch == index),
        )
    return ledger


def ledger_matches_result(ledger: Ledger, result: dict[str, Any]) -> dict[str, Any]:
    indices = ledger.committed_batch_indices()
    return {
        "committed_batches": indices,
        "expected_batches": len(result["batches"]),
        "complete": indices == list(range(len(result["batches"]))),
        "progress": ledger.last_progress(),
    }


def recover_to_completion(
    spec: Any,
    result: dict[str, Any],
    db_path: str | Path,
) -> tuple[Ledger, dict[str, Any]]:
    ledger = Ledger(db_path)
    progress = ledger.last_progress()
    last_batch = -1 if progress is None else int(progress["last_committed_batch_index"])
    unknown_batches = [last_batch + 1] if last_batch + 1 < len(result["batches"]) else []
    items_by_batch: dict[int, list[dict[str, Any]]] = {}
    for item in result["items"]:
        items_by_batch.setdefault(int(item["batch_index"]), []).append(item)
    for batch in result["batches"]:
        index = int(batch["batch_index"])
        if index <= last_batch:
            continue
        payload = schedule_payload_for_batch(index, batch, spec)
        cursor = {
            "integer_format_version": INTEGER_FORMAT_VERSION,
            "last_committed_batch_index": index,
            "last_committed_event_seq": int(batch["end_seq"]),
            "next_event_seq": int(batch["end_seq"]) + 1,
            "summary_content_digest": result["summary"]["content_digest"],
        }
        ledger.commit_batch(batch, items_by_batch[index], payload, cursor)
    report = {
        "last_committed_batch_before_recovery": last_batch,
        "unknown_batches_marked_and_rerun": unknown_batches,
        "committed_batch_indices_after_recovery": ledger.committed_batch_indices(),
        "progress_after_recovery": ledger.last_progress(),
    }
    return ledger, report
