"""Deterministic request-to-batch planning.

Stable rules (documented and exercised by tests):

1. Requests are ordered by ``(arrival, request_id)``.
2. Workers are ordered by ``worker_id``; consecutive requests are handed to
   workers round-robin in that order, regardless of dict iteration or
   dispatch order.
3. Each worker cuts its own assigned stream into batches of exactly
   ``worker.batch_size`` (the final one may be smaller); the batch sequence
   number starts at zero.
4. The batch id is
   ``B:<worker_id>:<batch_seq>:<short_digest>`` where the digest input is the
   canonical JSON object
   ``{"arrivals": [...], "batch_seq": n, "worker_id": w}``.

Because every step is total-ordered and every id is a hash of canonical
bytes, the plan and every batch id are byte identical across runs.
"""

from dataclasses import dataclass, field

from .canonical import dump_bytes, short_digest


@dataclass(frozen=True)
class PlannedBatch:
    batch_id: str
    worker_id: str
    batch_seq: int
    requests: tuple          # tuple of request ids, rank order
    arrivals: tuple          # parallel arrival sequence numbers
    vectors: dict            # request_id -> vector tuple
    batch_size: int

    def rank_of(self, request_id):
        return self.requests.index(request_id)


@dataclass
class Plan:
    batches: list = field(default_factory=list)

    def by_worker(self):
        grouped = {}
        for batch in self.batches:
            grouped.setdefault(batch.worker_id, []).append(batch)
        return grouped

    def batch(self, batch_id):
        for batch in self.batches:
            if batch.batch_id == batch_id:
                return batch
        return None

    @property
    def batch_sizes(self):
        return {b.batch_id: len(b.requests) for b in self.batches}


def make_batch_id(worker_id, batch_seq, arrivals):
    payload = {
        "worker_id": worker_id,
        "batch_seq": batch_seq,
        "arrivals": list(arrivals),
    }
    return f"B:{worker_id}:{batch_seq}:{short_digest(payload, 12)}"


def id_rule_bytes(worker_id, batch_seq, arrivals) -> bytes:
    """Expose the exact canonical bytes fed into the batch id hash."""
    return dump_bytes({
        "worker_id": worker_id,
        "batch_seq": batch_seq,
        "arrivals": list(arrivals),
    })


def build_plan(spec) -> Plan:
    ordered_requests = sorted(spec.requests,
                              key=lambda r: (r.arrival, r.request_id))
    ordered_workers = sorted(spec.workers, key=lambda w: w.worker_id)
    worker_buckets = {w.worker_id: [] for w in ordered_workers}
    for index, request in enumerate(ordered_requests):
        worker = ordered_workers[index % len(ordered_workers)]
        worker_buckets[worker.worker_id].append(request)

    batches = []
    for worker in ordered_workers:
        stream = worker_buckets[worker.worker_id]
        batch_seq = 0
        for start in range(0, len(stream), worker.batch_size):
            members = stream[start:start + worker.batch_size]
            members = sorted(members, key=lambda r: (r.arrival, r.request_id))
            arrivals = tuple(r.arrival for r in members)
            batch_id = make_batch_id(worker.worker_id, batch_seq, arrivals)
            batches.append(PlannedBatch(
                batch_id=batch_id,
                worker_id=worker.worker_id,
                batch_seq=batch_seq,
                requests=tuple(r.request_id for r in members),
                arrivals=arrivals,
                vectors={r.request_id: r.vector for r in members},
                batch_size=worker.batch_size,
            ))
            batch_seq += 1
    return Plan(batches=batches)
