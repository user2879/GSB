"""Top-level orchestration: build, schedule, persist, recover and replay."""

import os
import signal
from dataclasses import dataclass, field

from .invalidation import (InvalidationReport, propagate, recompute_all,
                           recompute_invalid, results_to_bytes)
from .ledger import Ledger, open_ledger, replay_check
from .planner import build_plan
from .receipts import encode_receipt
from .scheduler import schedule_all


class CrashForRecovery(Exception):
    """Raised after a forced-termination simulation so no exit report runs."""


@dataclass
class RunOutcome:
    plan: object
    results: dict
    ledger_path: str
    trace: object = None
    report: InvalidationReport = None
    receipts: dict = field(default_factory=dict)
    recovered: dict = None


def synthesize_receipt(result, batch, param_version, worker_id, fmt="r3",
                       duration_ms=0, duplicate_map=None):
    """Build receipts in any historical format from a computed batch."""
    duplicate_map = duplicate_map or {}
    records = []
    for rank, request_id in enumerate(batch.requests):
        normalized = {
            "batch_id": batch.batch_id,
            "request_id": request_id,
            "output_digest": result.output_digest,
            "worker_id": worker_id,
            "param_version": param_version,
            "duration_ms": duration_ms,
            "rank": rank,
            "duplicate_ref": duplicate_map.get(request_id),
            "param_digest": result.param_digest,
        }
        payload = encode_receipt(normalized, fmt)
        from .receipts import decode_receipt
        record = decode_receipt(payload, known_request_ids=set(batch.requests),
                                batch_sizes={batch.batch_id:
                                             len(batch.requests)})
        records.append(record)
    return records


def run_full(spec, ledger_path, use_scheduler=False, receipt_fmt="r3",
             duplicate_map=None):
    """Full rebuild from an empty spec; returns canonical result bytes."""
    plan = build_plan(spec)
    if use_scheduler:
        results, trace = schedule_all(spec, plan)
    else:
        results = recompute_all(spec, plan)
        trace = None
    ledger = Ledger(ledger_path, replace=True)
    receipts = {}
    for batch in sorted(plan.batches,
                        key=lambda b: (b.worker_id, b.batch_seq)):
        result = results[batch.batch_id]
        worker = spec.worker(batch.worker_id)
        batch_receipts = synthesize_receipt(
            result, batch, worker.param_version, worker.worker_id,
            fmt=receipt_fmt, duplicate_map=duplicate_map)
        receipts[batch.batch_id] = batch_receipts
        ledger.commit_batch(result, receipts=batch_receipts)
    ledger.close()
    return plan, results, receipts, trace


def run_incremental(old_spec, new_spec, ledger_path, receipt_fmt="r3",
                    duplicate_map=None):
    """Recompute only invalidated batches; keep valid ledger rows untouched."""
    old_plan = build_plan(old_spec)
    new_plan = build_plan(new_spec)
    report = propagate(old_spec, new_spec, old_plan, new_plan)
    ledger, status = open_ledger(ledger_path)
    ledger.append_traces(report.traces)

    invalid_ids = {row["batch_id"] for row in report.invalid_batches}
    fresh = recompute_invalid(new_spec, new_plan, invalid_ids)
    for batch in sorted(new_plan.batches,
                        key=lambda b: (b.worker_id, b.batch_seq)):
        if batch.batch_id not in invalid_ids:
            continue
        result = fresh[batch.batch_id]
        worker = new_spec.worker(batch.worker_id)
        batch_receipts = synthesize_receipt(
            result, batch, worker.param_version, worker.worker_id,
            fmt=receipt_fmt, duplicate_map=duplicate_map)
        ledger.commit_batch(result, receipts=batch_receipts,
                            traces=[t for t in report.traces
                                    if t.get("batch_id") == batch.batch_id])
    ledger.close()
    merged = recompute_all(old_spec, old_plan)
    merged.update(fresh)
    return new_plan, merged, report, status


def simulate_force_terminate(spec, ledger_path, crash_after,
                             receipt_fmt="r3", duplicate_map=None):
    """Commit *crash_after* batches then kill the process mid-batch.

    The current batch is deliberately never committed, proving that only
    fully committed batch rows can be used during recovery.
    """
    plan = build_plan(spec)
    results = recompute_all(spec, plan)
    ledger = Ledger(ledger_path, replace=True)
    ordered = sorted(plan.batches, key=lambda b: (b.worker_id, b.batch_seq))
    for index, batch in enumerate(ordered):
        if index >= crash_after:
            ledger.close()
            os.kill(os.getpid(), signal.SIGKILL)
        result = results[batch.batch_id]
        worker = spec.worker(batch.worker_id)
        batch_receipts = synthesize_receipt(
            result, batch, worker.param_version, worker.worker_id,
            fmt=receipt_fmt, duplicate_map=duplicate_map)
        ledger.commit_batch(result, receipts=batch_receipts)


def recover(spec, ledger_path):
    """Recover from a forced termination.

    Only fully committed batches are trusted. Planned batches missing from
    the ledger are marked ``unknown`` and scheduled for rerun; no exit
    report is fabricated for them.
    """
    plan = build_plan(spec)
    ledger, status = open_ledger(ledger_path)
    committed = set(ledger.batch_ids())
    unknown_batches = [
        {"batch_id": b.batch_id, "worker_id": b.worker_id,
         "batch_seq": b.batch_seq, "requests": list(b.requests),
         "state": "unknown", "action": "rerun_required"}
        for b in sorted(plan.batches, key=lambda x: (x.worker_id, x.batch_seq))
        if b.batch_id not in committed
    ]
    results = recompute_all(spec, plan)
    replay_check(ledger, plan, results)
    for batch in sorted(plan.batches, key=lambda b: (b.worker_id, b.batch_seq)):
        if batch.batch_id in committed:
            continue
        worker = spec.worker(batch.worker_id)
        batch_receipts = synthesize_receipt(
            results[batch.batch_id], batch, worker.param_version,
            worker.worker_id)
        ledger.commit_batch(results[batch.batch_id],
                            receipts=batch_receipts)
    ledger.close()
    return {
        "status": status,
        "unknown_batches": unknown_batches,
        "committed_batch_ids": sorted(committed),
        "replayed_ok": True,
    }


def verify_byte_identical(spec_a, spec_b, path_a, path_b, use_scheduler=False):
    """Compare two independent runs byte for byte via canonical result bytes."""
    plan_a, results_a, _, _ = run_full(spec_a, path_a,
                                       use_scheduler=use_scheduler)
    plan_b, results_b, _, _ = run_full(spec_b, path_b,
                                       use_scheduler=use_scheduler)
    bytes_a = results_to_bytes(results_a)
    bytes_b = results_to_bytes(results_b)
    return {
        "identical": bytes_a == bytes_b,
        "byte_length": len(bytes_a),
        "digest_a": __import__("hashlib").sha256(bytes_a).hexdigest(),
        "digest_b": __import__("hashlib").sha256(bytes_b).hexdigest(),
        "batch_ids_a": [b.batch_id for b in plan_a.batches],
        "batch_ids_b": [b.batch_id for b in plan_b.batches],
    }
