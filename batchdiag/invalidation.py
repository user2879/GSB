"""Invalidation propagation along worker-param-version -> batch -> request.

The dependency chain is explicit:

    worker declaration (seed / dims / batch_size / param_version)
        -> batches owned by that worker
            -> request attribution rows inside those batches

A worker-level change directly invalidates every batch of that worker; each
invalidated batch transitively invalidates the request attribution rows it
contains. Nothing outside the affected worker is touched, and because batch
ids remain stable when only weight-driving fields change, surviving batch
receipts stay byte identical.
"""

from dataclasses import dataclass

from .canonical import dump_bytes
from .model import run_batch
from .planner import build_plan

_CHAIN_FIELDS = ("seed", "model_dim", "hidden_dim", "batch_size",
                 "param_version", "port", "quota")


@dataclass
class InvalidationReport:
    changed_workers: list
    removed_workers: list
    added_workers: list
    invalid_batches: list          # list of dict, stable order
    invalid_requests: list         # list of dict, stable order
    valid_batch_ids: list
    traces: list                   # propagation trail rows
    field_diffs: list              # worker-level field differences


def _worker_map(workers):
    return {w.worker_id: w for w in workers}


def diff_workers(old_spec, new_spec):
    """Return stable per-worker field differences (first field ordered)."""
    old_map = _worker_map(old_spec.workers)
    new_map = _worker_map(new_spec.workers)
    diffs = []
    for worker_id in sorted(set(old_map) | set(new_map)):
        old = old_map.get(worker_id)
        new = new_map.get(worker_id)
        if old is None:
            diffs.append({"worker_id": worker_id, "kind": "added",
                          "changed_fields": list(_CHAIN_FIELDS)})
            continue
        if new is None:
            diffs.append({"worker_id": worker_id, "kind": "removed",
                          "changed_fields": list(_CHAIN_FIELDS)})
            continue
        changed = [name for name in _CHAIN_FIELDS
                   if getattr(old, name) != getattr(new, name)]
        if changed:
            diffs.append({
                "worker_id": worker_id,
                "kind": "modified",
                "changed_fields": changed,
                "old": {name: getattr(old, name) for name in changed},
                "new": {name: getattr(new, name) for name in changed},
            })
    return diffs


def propagate(old_spec, new_spec, old_plan=None, new_plan=None,
              old_results=None):
    """Compute the invalidation report between two specifications."""
    old_plan = old_plan or build_plan(old_spec)
    new_plan = new_plan or build_plan(new_spec)
    diffs = diff_workers(old_spec, new_spec)

    changed = sorted(d["worker_id"] for d in diffs
                     if d["kind"] == "modified")
    removed = sorted(d["worker_id"] for d in diffs
                     if d["kind"] == "removed")
    added = sorted(d["worker_id"] for d in diffs
                   if d["kind"] == "added")
    affected_workers = set(changed) | set(removed)

    invalid_batches = []
    invalid_requests = []
    traces = []
    for batch in sorted(old_plan.batches,
                        key=lambda b: (b.worker_id, b.batch_seq)):
        if batch.worker_id not in affected_workers:
            continue
        reasons = [d for d in diffs if d["worker_id"] == batch.worker_id]
        reason_fields = sorted({
            field_name
            for d in reasons
            for field_name in d["changed_fields"]
        })
        invalid_batches.append({
            "batch_id": batch.batch_id,
            "worker_id": batch.worker_id,
            "batch_seq": batch.batch_seq,
            "direct": True,
            "reason_fields": reason_fields,
        })
        traces.append({
            "level": "worker->batch",
            "worker_id": batch.worker_id,
            "batch_id": batch.batch_id,
            "request_id": None,
            "reason_fields": reason_fields,
        })
        for rank, request_id in enumerate(batch.requests):
            invalid_requests.append({
                "batch_id": batch.batch_id,
                "worker_id": batch.worker_id,
                "request_id": request_id,
                "rank": rank,
            })
            traces.append({
                "level": "batch->request",
                "worker_id": batch.worker_id,
                "batch_id": batch.batch_id,
                "request_id": request_id,
                "reason_fields": reason_fields,
            })

    for worker_id in added:
        traces.append({
            "level": "worker->batch",
            "worker_id": worker_id,
            "batch_id": None,
            "request_id": None,
            "reason_fields": list(_CHAIN_FIELDS),
        })

    invalid_ids = {row["batch_id"] for row in invalid_batches}
    valid_batch_ids = sorted(b.batch_id for b in old_plan.batches
                             if b.batch_id not in invalid_ids)
    return InvalidationReport(
        changed_workers=changed,
        removed_workers=removed,
        added_workers=added,
        invalid_batches=invalid_batches,
        invalid_requests=invalid_requests,
        valid_batch_ids=valid_batch_ids,
        traces=traces,
        field_diffs=diffs,
    )


def recompute_invalid(spec, plan, invalid_batch_ids):
    """Recompute only the given invalidated batches, deterministically."""
    invalid_set = set(invalid_batch_ids)
    worker_map = _worker_map(spec.workers)
    results = {}
    for batch in sorted(plan.batches, key=lambda b: (b.worker_id, b.batch_seq)):
        if batch.batch_id not in invalid_set:
            continue
        worker = worker_map[batch.worker_id]
        results[batch.batch_id] = run_batch(worker, batch)
    return results


def recompute_all(spec, plan=None):
    plan = plan or build_plan(spec)
    return recompute_invalid(spec, plan, {b.batch_id for b in plan.batches})


def results_to_bytes(results):
    """Canonical byte rendering used for incremental vs full comparison."""
    ordered = sorted(results, key=lambda batch_id: batch_id)
    payload = [
        {
            "batch_id": batch_id,
            "worker_id": results[batch_id].worker_id,
            "param_version": results[batch_id].param_version,
            "param_digest": results[batch_id].param_digest,
            "rank_order": list(results[batch_id].rank_order),
            "sorted_requests": list(results[batch_id].sorted_requests),
            "per_request_digest": results[batch_id].per_request_digest,
            "output_digest": results[batch_id].output_digest,
            "basis": results[batch_id].basis,
        }
        for batch_id in ordered
    ]
    return dump_bytes(payload)
