"""Deterministic batch planning.

Requests are sorted by arrival sequence and workers by worker id. Each round,
every worker (in id order) consumes up to its own ``batch_size`` requests.
Identifiers are derived from an explicit stable serialization of the worker id,
batch index and participating arrival sequences, so the plan is byte-identical
regardless of dispatch order or dictionary iteration order.

Remainder rule (explicit): when fewer than ``batch_size`` requests remain for a
worker turn, those requests still form one trailing partial batch carrying
``remainder=True`` and a human-readable rationale; requests are never dropped or
re-ordered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .canonical import canonical_json, short_digest
from .spec import SystemSpec, WorkerSpec

REMAINDER_RATIONALE = (
    "fewer than batch_size requests remain; emit one trailing partial batch "
    "without dropping or reordering requests"
)


@dataclass(frozen=True)
class Batch:
    batch_id: str
    worker_id: str
    batch_index: int
    request_ids: tuple[str, ...]
    arrival_seqs: tuple[int, ...]
    remainder: bool
    rationale: str

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "batch_index": self.batch_index,
            "request_ids": list(self.request_ids),
            "arrival_seqs": list(self.arrival_seqs),
            "remainder": self.remainder,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class BatchPlan:
    batches: tuple[Batch, ...]
    remainder_rule: str
    request_to_batch: dict[str, str] = field(default_factory=dict)

    def batches_for_worker(self, worker_id: str) -> tuple[Batch, ...]:
        return tuple(batch for batch in self.batches if batch.worker_id == worker_id)


def build_batch_identity(worker: WorkerSpec, batch_index: int, arrival_seqs: tuple[int, ...]) -> str:
    """Derive the batch id via the explicit stable serialization rule."""

    payload = {
        "worker_id": worker.worker_id,
        "batch_index": batch_index,
        "arrival_seqs": list(arrival_seqs),
    }
    return f"batch-{worker.worker_id}-{batch_index:04d}-{short_digest(payload, 12)}"


def build_plan(spec: SystemSpec) -> BatchPlan:
    """Split the request stream into deterministic, worker-quota batches."""

    ordered_requests = sorted(spec.requests, key=lambda request: request.arrival_seq)
    ordered_workers = sorted(spec.workers, key=lambda worker: worker.worker_id)
    request_by_id = {request.request_id: request for request in ordered_requests}

    queue = list(ordered_requests)
    per_worker_counters = {worker.worker_id: 0 for worker in ordered_workers}
    batches: list[Batch] = []
    request_to_batch: dict[str, str] = {}

    cursor = 0
    while cursor < len(queue):
        progressed = False
        for worker in ordered_workers:
            if cursor >= len(queue):
                break
            chunk = queue[cursor : cursor + worker.batch_size]
            cursor += len(chunk)
            progressed = True
            batch_index = per_worker_counters[worker.worker_id]
            per_worker_counters[worker.worker_id] += 1
            request_ids = tuple(request.request_id for request in chunk)
            arrival_seqs = tuple(request.arrival_seq for request in chunk)
            remainder = len(chunk) < worker.batch_size
            rationale = REMAINDER_RATIONALE if remainder else "full batch of batch_size requests"
            batch_id = build_batch_identity(worker, batch_index, arrival_seqs)
            batch = Batch(
                batch_id=batch_id,
                worker_id=worker.worker_id,
                batch_index=batch_index,
                request_ids=request_ids,
                arrival_seqs=arrival_seqs,
                remainder=remainder,
                rationale=rationale,
            )
            batches.append(batch)
            for request_id in request_ids:
                request_to_batch[request_id] = batch_id
        if not progressed:  # pragma: no cover - defensive guard
            break

    # Touch request_by_id to keep the lookup contract explicit for integrators.
    _ = request_by_id
    _ = canonical_json  # re-exported stability primitive for external callers
    return BatchPlan(
        batches=tuple(batches),
        remainder_rule=REMAINDER_RATIONALE,
        request_to_batch=dict(sorted(request_to_batch.items())),
    )
