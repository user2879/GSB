"""Orchestration: plan -> loopback workers -> normalized receipts -> ledger."""

from __future__ import annotations

from dataclasses import dataclass

from .inference import ForwardResult, build_parameters, forward_batch
from .invalidation import InvalidationReport
from .ledger import SnapshotLedger
from .network import WorkerClient, WorkerService, check_port_bindable
from .planning import Batch, BatchPlan, build_plan
from .receipt import decode_receipt, validate_chain
from .scheduler import Scheduler
from .spec import RequestSpec, SystemSpec, WorkerSpec


def _reconstruct_worker(payload: dict) -> WorkerSpec:
    return WorkerSpec(**payload["worker"])


def _reconstruct_requests(payload: dict) -> list[RequestSpec]:
    return [RequestSpec(**item) for item in payload["requests"]]


def worker_compute(payload: dict) -> dict:
    """Worker-side deterministic computation reachable only via loopback."""

    worker = _reconstruct_worker(payload)
    requests = _reconstruct_requests(payload)
    spec = SystemSpec(
        requests=tuple(requests),
        workers=(worker,),
        budget=None,  # type: ignore[arg-type]
    )
    batch = Batch(**payload["batch"])
    result = forward_batch(spec, worker, batch, build_parameters(worker))
    ordered_ids = sorted(batch.request_ids)
    receipts = []
    for rank, request_id in enumerate(ordered_ids):
        receipts.append(
            {
                "format_version": "r3",
                "batch_id": batch.batch_id,
                "request_id": request_id,
                "output_digest": result.per_request[request_id],
                "param_version": worker.param_version,
                "elapsed_ms": 0,
                "worker_id": worker.worker_id,
                "rank": rank,
                "duplicate_ref": None,
                "param_digest": result.param_digest,
            }
        )
    return {
        "batch_id": result.batch_id,
        "batch_index": batch.batch_index,
        "worker_id": result.worker_id,
        "intermediate_dim": result.intermediate_dim,
        "param_version": result.param_version,
        "param_digest": result.param_digest,
        "output_digest": result.output_digest,
        "per_request": result.per_request,
        "receipts": receipts,
    }


@dataclass
class EngineRunSummary:
    processed_batches: tuple[str, ...]
    skipped_batches: tuple[str, ...]
    peak_global: int
    peak_worker: dict[str, int]
    total_budget_ms: int


class DiagnosticEngine:
    def __init__(self, spec: SystemSpec, ledger: SnapshotLedger) -> None:
        self.spec = spec
        self.ledger = ledger
        self.plan: BatchPlan = build_plan(spec)
        self.services: list[WorkerService] = []

    # ------------------------------------------------------------- lifecycle
    def start_workers(self) -> None:
        for worker in sorted(self.spec.workers, key=lambda item: item.worker_id):
            check_port_bindable(worker.host, worker.port)
        for worker in sorted(self.spec.workers, key=lambda item: item.worker_id):
            service = WorkerService(worker.host, worker.port, worker_compute)
            service.start()
            self.services.append(service)

    def stop_workers(self) -> None:
        for service in self.services:
            service.stop()
        self.services.clear()

    # ----------------------------------------------------------------- jobs
    def _job_for(self, batch: Batch) -> dict:
        worker = self.spec.worker_by_id()[batch.worker_id]
        request_by_id = self.spec.request_by_id()
        return {
            "batch_index": batch.batch_index,
            "batch_id": batch.batch_id,
            "worker_id": worker.worker_id,
            "worker": worker.__dict__,
            "batch": batch.to_dict(),
            "requests": [
                request_by_id[request_id].__dict__ for request_id in batch.request_ids
            ],
        }

    # ------------------------------------------------------------- execution
    def run(
        self,
        *,
        only_batch_ids: set[str] | None = None,
        invalidation: InvalidationReport | None = None,
        client_factory=WorkerClient,
    ) -> EngineRunSummary:
        """Process batches; skip batches already committed unless invalidated."""

        targets = [
            batch
            for batch in sorted(
                self.plan.batches, key=lambda item: (item.worker_id, item.batch_index)
            )
            if only_batch_ids is None or batch.batch_id in only_batch_ids
        ]
        already_committed = set(self.ledger.committed_batch_ids())
        jobs: list[dict] = []
        skipped: list[str] = []
        for batch in targets:
            if batch.batch_id in already_committed and only_batch_ids is None:
                skipped.append(batch.batch_id)
                continue
            jobs.append(self._job_for(batch))

        scheduler = Scheduler(self.spec, client_factory)
        scheduled = scheduler.run(jobs) if jobs else []
        known_ids = {request.request_id for request in self.spec.requests}
        processed: list[str] = []

        for scheduled_result in scheduled:
            batch = next(item for item in self.plan.batches if item.batch_id == scheduled_result.batch_id)
            worker = self.spec.worker_by_id()[batch.worker_id]
            local_result: ForwardResult = forward_batch(self.spec, worker, batch)
            response = scheduled_result.response
            if response["output_digest"] != local_result.output_digest:
                raise RuntimeError(
                    f"worker/local output digest mismatch for batch {batch.batch_id}"
                )
            normalized = [
                decode_receipt(raw, known_request_ids=known_ids)
                for raw in response["receipts"]
            ]
            validate_chain(normalized)
            steps = invalidation.steps if invalidation is not None else tuple()
            steps = [step for step in steps if step.subject == batch.batch_id or
                     step.subject in batch.request_ids or
                     (step.level == "worker_param_version" and step.worker_id == batch.worker_id)]
            self.ledger.commit_batch(
                batch=batch,
                forward_result=local_result,
                normalized_receipts=normalized,
                invalidation_steps=steps,
                timing={
                    "elapsed_ms": scheduled_result.timing.elapsed_ms,
                    "wait_started_ms": scheduled_result.timing.wait_started_ms,
                    "wait_ended_ms": scheduled_result.timing.wait_ended_ms,
                    "attempts": scheduled_result.timing.attempts,
                },
            )
            processed.append(batch.batch_id)

        return EngineRunSummary(
            processed_batches=tuple(processed),
            skipped_batches=tuple(skipped),
            peak_global=scheduler.peak_global,
            peak_worker=dict(sorted(scheduler.peak_worker.items())),
            total_budget_ms=self.ledger.total_budget_ms(),
        )
