"""Deterministic worker assignment and batch planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .jsonio import canonical_bytes, sha256_bytes
from .model import parameter_digest
from .spec import RequestSet, RequestSpec, WorkerSet, WorkerSpec


PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    worker_id: str
    batch_seq: int
    request_ids: tuple[str, ...]
    arrival_seqs: tuple[int, ...]
    param_version_seq: int
    param_digest: str
    request_binding_digest: str
    plan_digest: str

    def rank_by_request_id(self) -> dict[str, int]:
        return {
            request_id: index for index, request_id in enumerate(self.request_ids)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "worker_id": self.worker_id,
            "batch_seq": self.batch_seq,
            "request_ids": list(self.request_ids),
            "arrival_seqs": list(self.arrival_seqs),
            "param_version_seq": self.param_version_seq,
            "param_digest": self.param_digest,
            "request_binding_digest": self.request_binding_digest,
            "plan_digest": self.plan_digest,
        }


@dataclass(frozen=True)
class DispatchPlan:
    schema_version: int
    worker_set_version: int
    batches: tuple[BatchPlan, ...]
    content_digest: str

    def by_id(self) -> dict[str, BatchPlan]:
        return {batch.batch_id: batch for batch in self.batches}

    def sorted_batches(self) -> tuple[BatchPlan, ...]:
        return tuple(
            sorted(self.batches, key=lambda item: (item.worker_id, item.batch_seq, item.batch_id))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "worker_set_version": self.worker_set_version,
            "batches": [batch.to_dict() for batch in self.sorted_batches()],
            "content_digest": self.content_digest,
        }


def batch_id_for(worker_id: str, batch_seq: int, arrival_seqs: tuple[int, ...]) -> str:
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "worker_id": worker_id,
        "batch_seq": batch_seq,
        "arrival_seqs": list(arrival_seqs),
    }
    return "batch:" + sha256_bytes(canonical_bytes(payload))


def _request_binding_digest(requests: list[RequestSpec]) -> str:
    payload = [
        {
            "request_id": request.request_id,
            "arrival_seq": request.arrival_seq,
            "worker_id": request.worker_id,
            "input": list(request.input),
        }
        for request in sorted(requests, key=lambda item: item.arrival_seq)
    ]
    return sha256_bytes(canonical_bytes(payload))


def _batch_plan_digest(
    *,
    batch_id: str,
    worker: WorkerSpec,
    param_digest: str,
    request_binding_digest: str,
) -> str:
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "batch_id": batch_id,
        "worker": worker.to_dict(),
        "param_digest": param_digest,
        "request_binding_digest": request_binding_digest,
    }
    return sha256_bytes(canonical_bytes(payload))


def build_plan(worker_set: WorkerSet, request_set: RequestSet) -> DispatchPlan:
    worker_map = worker_set.by_id()
    grouped: dict[str, list[RequestSpec]] = {worker_id: [] for worker_id in worker_map}
    for request in sorted(request_set.requests, key=lambda item: item.arrival_seq):
        grouped[request.worker_id].append(request)

    batches: list[BatchPlan] = []
    for worker_id in sorted(grouped):
        worker = worker_map[worker_id]
        requests = grouped[worker_id]
        param_digest = parameter_digest(worker)
        for batch_seq, start in enumerate(range(0, len(requests), worker.batch_size)):
            members = requests[start : start + worker.batch_size]
            arrival_seqs = tuple(member.arrival_seq for member in members)
            batch_id = batch_id_for(worker_id, batch_seq, arrival_seqs)
            binding_digest = _request_binding_digest(members)
            batches.append(
                BatchPlan(
                    batch_id=batch_id,
                    worker_id=worker_id,
                    batch_seq=batch_seq,
                    request_ids=tuple(member.request_id for member in members),
                    arrival_seqs=arrival_seqs,
                    param_version_seq=worker.param_version_seq,
                    param_digest=param_digest,
                    request_binding_digest=binding_digest,
                    plan_digest=_batch_plan_digest(
                        batch_id=batch_id,
                        worker=worker,
                        param_digest=param_digest,
                        request_binding_digest=binding_digest,
                    ),
                )
            )
    ordered = tuple(
        sorted(batches, key=lambda item: (item.worker_id, item.batch_seq, item.batch_id))
    )
    content = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "worker_set_version": worker_set.worker_set_version,
        "batches": [batch.to_dict() for batch in ordered],
    }
    return DispatchPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        worker_set_version=worker_set.worker_set_version,
        batches=ordered,
        content_digest=sha256_bytes(canonical_bytes(content)),
    )
