"""Deterministic batch planning.

Requests are grouped per worker, ordered by arrival sequence, then split into
fixed-size batches.  All identifiers and content digests are derived through
explicit stable (canonical, key-sorted) serialization, so the same request
stream yields byte-identical plans regardless of dispatch order or dict
traversal order.
"""

from dataclasses import dataclass

from .digest import canonical_json, digest
from .errors import SpecError


@dataclass(frozen=True)
class Batch:
    batch_id: str
    worker_id: str
    batch_index: int
    request_ids: tuple
    arrivals: tuple
    content_digest: str

    def to_dict(self):
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "batch_index": self.batch_index,
            "request_ids": list(self.request_ids),
            "arrivals": list(self.arrivals),
            "content_digest": self.content_digest,
        }


def _batch_id(worker_id, batch_index, arrivals):
    payload = {
            "kind": "batch_id",
            "worker_id": worker_id,
            "batch_index": batch_index,
            "arrivals": list(arrivals),
        }
    return f"{worker_id}-b{batch_index:04d}-{digest(payload)[:12]}"


def _content_digest(worker_id, batch_index, request_ids, arrivals,
                    batch_size, seed):
    return digest(
        {
            "kind": "batch_content",
            "worker_id": worker_id,
            "batch_index": batch_index,
            "batch_size": int(batch_size),
            "seed": int(seed),
            "members": [
                {"arrival": arrival, "request_id": request_id}
                for arrival, request_id in sorted(zip(arrivals, request_ids))
            ],
        }
    )


def build_plan(spec, batch_size, seed):
    """Split the request stream into deterministic batches.

    Worker quota (``slot_cap``) bounds how many batches a worker may own.
    Returns a :class:`Plan` whose ``signature`` pins the entire partition.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) \
            or batch_size <= 0:
        raise SpecError("batch_size must be a positive integer",
                        field="batch_size")

    grouped = {}
    for request in spec.requests:
        grouped.setdefault(request.worker_id, []).append(request)

    batches = []
    # Sorted worker iteration removes any dependence on dict ordering.
    for worker_id in sorted(grouped):
        ordered = sorted(grouped[worker_id], key=lambda r: r.arrival)
        worker = spec.worker_by_id[worker_id]
        raw_chunks = [
            ordered[i:i + batch_size]
            for i in range(0, len(ordered), batch_size)
        ]
        if worker.slot_cap and len(raw_chunks) > worker.slot_cap:
            raise SpecError(
                f"worker {worker_id!r} needs {len(raw_chunks)} batches but "
                f"slot_cap quota is {worker.slot_cap}",
                field="slot_cap",
            )
        for batch_index, chunk in enumerate(raw_chunks):
            request_ids = tuple(r.request_id for r in chunk)
            arrivals = tuple(r.arrival for r in chunk)
            batches.append(
                Batch(
                    batch_id=_batch_id(worker_id, batch_index, arrivals),
                    worker_id=worker_id,
                    batch_index=batch_index,
                    request_ids=request_ids,
                    arrivals=arrivals,
                    content_digest=_content_digest(
                        worker_id, batch_index, request_ids, arrivals,
                        batch_size, seed,
                    ),
                )
            )

    batches = tuple(batches)
    signature = digest(
        {
            "kind": "plan_signature",
            "batch_size": int(batch_size),
            "seed": int(seed),
            "batches": [b.to_dict() for b in batches],
        }
    )
    return Plan(batch_size, seed, batches, signature)


class Plan:
    def __init__(self, batch_size, seed, batches, signature):
        self.batch_size = batch_size
        self.seed = seed
        self.batch_batches = batches
        self.signature = signature

    @property
    def batches(self):
        return self.batch_batches

    def by_worker(self):
        result = {}
        for batch in self.batches:
            result.setdefault(batch.worker_id, []).append(batch)
        return result

    def canonical_bytes(self):
        return canonical_json(
            {
                "batch_size": self.batch_size,
                "seed": self.seed,
                "signature": self.signature,
                "batches": [b.to_dict() for b in self.batches],
            }
        )

    def __len__(self):
        return len(self.batches)
