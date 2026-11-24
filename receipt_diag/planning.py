"""Deterministic batching and stable batch identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_bytes, canonical_json, digest_json, sha256_bytes
from .specs import RequestSpec, SampleSpec


BATCH_SCHEMA = "receipt-diag-batch-v1"


@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    worker_id: str
    batch_seq: int
    requests: tuple[RequestSpec, ...]

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(item.request_id for item in self.requests)

    @property
    def arrival_seqs(self) -> tuple[int, ...]:
        return tuple(item.arrival_seq for item in self.requests)

    def rank_of(self, request_id: str) -> int:
        return self.request_ids.index(request_id)

    def canonical_source(self) -> dict[str, Any]:
        return {
            "schema": BATCH_SCHEMA,
            "worker_id": self.worker_id,
            "batch_seq": self.batch_seq,
            "arrival_seqs": list(self.arrival_seqs),
        }

    def canonical_source_text(self) -> str:
        return canonical_json(self.canonical_source())

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "batch_seq": self.batch_seq,
            "request_ids": list(self.request_ids),
            "arrival_seqs": list(self.arrival_seqs),
            "canonical_source": self.canonical_source_text(),
        }


@dataclass(frozen=True)
class ExecutionPlan:
    batches: tuple[BatchPlan, ...]
    fingerprint: str

    @property
    def by_id(self) -> dict[str, BatchPlan]:
        return {batch.batch_id: batch for batch in self.batches}

    def sorted_batches(self) -> tuple[BatchPlan, ...]:
        return tuple(
            sorted(self.batches, key=lambda item: (item.worker_id, item.batch_seq, item.batch_id))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_schema": BATCH_SCHEMA,
            "fingerprint": self.fingerprint,
            "batches": [batch.to_dict() for batch in self.sorted_batches()],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


def make_batch_id(worker_id: str, batch_seq: int, arrival_seqs: Iterable[int]) -> str:
    payload = {
        "schema": BATCH_SCHEMA,
        "worker_id": worker_id,
        "batch_seq": batch_seq,
        "arrival_seqs": list(arrival_seqs),
    }
    return "B_" + sha256_bytes(canonical_bytes(payload))


def plan_batches(spec: SampleSpec) -> ExecutionPlan:
    """Partition every assignment deterministically.

    Requests are sorted by the globally unique arrival sequence. Worker and
    batch traversal are sorted explicitly. Input JSON object order and the
    caller's dispatch order therefore cannot affect any byte of the plan.
    """

    expected_workers = {worker_id for _, worker_id in spec.rules.assignments}
    request_by_id = spec.request_by_id
    grouped: dict[str, list[RequestSpec]] = {worker_id: [] for worker_id in expected_workers}
    for request_id, worker_id in spec.rules.assignments:
        grouped[worker_id].append(request_by_id[request_id])

    batches: list[BatchPlan] = []
    for worker in spec.workers_by_id:
        worker_requests = sorted(
            grouped.get(worker.worker_id, []),
            key=lambda item: (item.arrival_seq, item.request_id),
        )
        for offset in range(0, len(worker_requests), worker.batch_size):
            batch_seq = offset // worker.batch_size + 1
            members = tuple(worker_requests[offset : offset + worker.batch_size])
            arrivals = tuple(item.arrival_seq for item in members)
            batch_id = make_batch_id(worker.worker_id, batch_seq, arrivals)
            batches.append(
                BatchPlan(
                    batch_id=batch_id,
                    worker_id=worker.worker_id,
                    batch_seq=batch_seq,
                    requests=members,
                )
            )
    batches.sort(key=lambda item: (item.worker_id, item.batch_seq, item.batch_id))
    plan_payload = [batch.to_dict() for batch in batches]
    return ExecutionPlan(batches=tuple(batches), fingerprint=digest_json(plan_payload))


def batch_ids_sorted(batch_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(batch_ids)))
