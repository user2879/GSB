from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from .canonical import canonical_digest
from .errors import CoverageError
from .models import Batch, Plan, RequestSpec, SpecBundle, WorkerSpec


BATCH_ID_PREFIX = "rb1"


def batch_size_for(worker: WorkerSpec, bundle: SpecBundle | None = None, default_batch_size: int | None = None) -> int:
    if worker.batch_size is not None:
        return worker.batch_size
    if default_batch_size is not None:
        return default_batch_size
    if bundle is not None:
        return bundle.rules.default_batch_size
    return 2


def derive_batch_id(worker_id: str, batch_index: int, arrival_seqs: Iterable[int]) -> str:
    """Derive an ID from the documented stable material only."""

    material = {
        "arrival_seqs": [int(seq) for seq in arrival_seqs],
        "batch_index": int(batch_index),
        "worker_id": worker_id,
    }
    return f"{BATCH_ID_PREFIX}_{canonical_digest(material, 20)}"


def build_plan(bundle: SpecBundle) -> Plan:
    return build_plan_from_parts(bundle.workers, bundle.requests, bundle.rules.default_batch_size)


def build_plan_from_parts(
    workers: Iterable[WorkerSpec],
    requests: Iterable[RequestSpec],
    default_batch_size: int = 2,
) -> Plan:
    worker_list = sorted(workers, key=lambda worker: worker.worker_id)
    request_list = sorted(requests, key=lambda request: request.arrival_seq)
    worker_map = {worker.worker_id: worker for worker in worker_list}
    grouped: dict[str, list[RequestSpec]] = defaultdict(list)
    for request in request_list:
        if request.worker_id in worker_map:
            grouped[request.worker_id].append(request)

    batches: list[Batch] = []
    for worker in worker_list:
        worker_requests = sorted(grouped.get(worker.worker_id, []), key=lambda request: request.arrival_seq)
        size = batch_size_for(worker, default_batch_size=default_batch_size)
        for batch_index, start in enumerate(range(0, len(worker_requests), size)):
            members = worker_requests[start : start + size]
            arrival_seqs = tuple(request.arrival_seq for request in members)
            request_ids = tuple(request.request_id for request in members)
            batch_id = derive_batch_id(worker.worker_id, batch_index, arrival_seqs)
            stable_ranks = {request.request_id: rank for rank, request in enumerate(members)}
            batches.append(
                Batch(
                    batch_id=batch_id,
                    worker_id=worker.worker_id,
                    batch_index=batch_index,
                    arrival_seqs=arrival_seqs,
                    request_ids=request_ids,
                    stable_ranks=stable_ranks,
                )
            )
    batches.sort(key=lambda batch: (batch.worker_id, batch.batch_index, batch.arrival_seqs))
    return Plan(tuple(batches))


def verify_coverage(plan: Plan, requests: Iterable[RequestSpec]) -> None:
    expected_ids = {request.request_id for request in requests}
    assigned: list[str] = []
    for batch in plan.batches:
        assigned.extend(batch.request_ids)

    counts = Counter(assigned)
    overlap = sorted(request_id for request_id, count in counts.items() if count > 1)
    missing = sorted(expected_ids - counts.keys())
    extra = sorted(counts.keys() - expected_ids)

    if overlap:
        raise CoverageError(
            "存在重复覆盖：每个请求必须恰好归属一个批次",
            coverage="overlap",
            request_ids=overlap,
            code_name="E_COVER_OVERLAP",
        )
    if missing or extra:
        raise CoverageError(
            "存在遗漏或计划外请求",
            coverage="missing" if missing else "extra",
            missing_request_ids=missing,
            extra_request_ids=extra,
            code_name="E_COVER_MISSING",
        )
    if len(assigned) != len(expected_ids):
        raise CoverageError(
            "覆盖数量不自洽",
            coverage="count",
            assigned_count=len(assigned),
            expected_count=len(expected_ids),
        )


def assignment_map(plan: Plan) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for batch in plan.batches:
        for rank, request_id in enumerate(batch.request_ids):
            result[request_id] = {
                "batch_id": batch.batch_id,
                "worker_id": batch.worker_id,
                "rank": rank,
                "arrival_seq": batch.arrival_seqs[rank],
            }
    return dict(sorted(result.items()))
