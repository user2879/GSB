"""Invalidation propagation along worker parameter version -> batch -> request.

A worker becomes invalid when any parameter-defining field (seed, dimensions or
parameter version) changes, when the worker is added/removed, or when its batch
size changes (which reshapes batch membership). Invalidity then flows to every
batch owned by that worker and finally to every request attributed to those
batches. All outputs are ordered by worker id and batch index for stable,
byte-comparable results.
"""

from __future__ import annotations

from dataclasses import dataclass

from .planning import BatchPlan
from .spec import SystemSpec, WorkerSpec

PARAM_FIELDS = ("seed", "param_version", "input_dim", "hidden_dim")
BATCH_FIELD = "batch_size"


@dataclass(frozen=True)
class PropagationStep:
    level: str
    worker_id: str
    subject: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "worker_id": self.worker_id,
            "subject": self.subject,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InvalidationReport:
    invalid_workers: tuple[str, ...]
    invalid_batch_ids: tuple[str, ...]
    invalid_request_ids: tuple[str, ...]
    steps: tuple[PropagationStep, ...]
    worker_reasons: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict:
        return {
            "invalid_workers": list(self.invalid_workers),
            "invalid_batch_ids": list(self.invalid_batch_ids),
            "invalid_request_ids": list(self.invalid_request_ids),
            "steps": [step.to_dict() for step in self.steps],
            "worker_reasons": {
                worker: list(reasons) for worker, reasons in sorted(self.worker_reasons.items())
            },
        }


def diff_workers(old: SystemSpec, new: SystemSpec) -> dict[str, tuple[str, ...]]:
    """Return ordered per-worker change reasons between two specifications."""

    old_workers = {worker.worker_id: worker for worker in old.workers}
    new_workers = {worker.worker_id: worker for worker in new.workers}
    reasons: dict[str, list[str]] = {}
    for worker_id in sorted(set(old_workers) | set(new_workers)):
        local: list[str] = []
        if worker_id not in old_workers:
            local.append("worker added in new specification")
            reasons[worker_id] = local
            continue
        if worker_id not in new_workers:
            local.append("worker removed from new specification")
            reasons[worker_id] = local
            continue
        old_worker: WorkerSpec = old_workers[worker_id]
        new_worker: WorkerSpec = new_workers[worker_id]
        for field_name in PARAM_FIELDS:
            if getattr(old_worker, field_name) != getattr(new_worker, field_name):
                local.append(
                    f"parameter field {field_name} changed: "
                    f"{getattr(old_worker, field_name)!r} -> {getattr(new_worker, field_name)!r}"
                )
        if old_worker.batch_size != new_worker.batch_size:
            local.append(
                f"{BATCH_FIELD} changed: {old_worker.batch_size} -> {new_worker.batch_size}"
            )
        if old_worker.port != new_worker.port or old_worker.host != new_worker.host:
            local.append(
                f"loopback endpoint changed: {old_worker.host}:{old_worker.port} "
                f"-> {new_worker.host}:{new_worker.port}"
            )
        if local:
            reasons[worker_id] = local
    return {worker: tuple(items) for worker, items in sorted(reasons.items())}


def propagate(old: SystemSpec, new: SystemSpec, plan: BatchPlan) -> InvalidationReport:
    """Propagate specification changes to invalid batches and requests."""

    worker_reasons = diff_workers(old, new)
    invalid_workers = tuple(worker_reasons)
    steps: list[PropagationStep] = []
    invalid_batches: list[str] = []
    invalid_requests: set[str] = set()

    plan_by_worker = {
        worker_id: plan.batches_for_worker(worker_id) for worker_id in invalid_workers
    }
    for worker_id in invalid_workers:
        for reason in worker_reasons[worker_id]:
            steps.append(
                PropagationStep(
                    level="worker_param_version",
                    worker_id=worker_id,
                    subject=worker_id,
                    reason=reason,
                )
            )
        worker_batches = sorted(plan_by_worker.get(worker_id, ()), key=lambda batch: batch.batch_index)
        for batch in worker_batches:
            steps.append(
                PropagationStep(
                    level="batch",
                    worker_id=worker_id,
                    subject=batch.batch_id,
                    reason=(
                        f"batch owned by invalid worker {worker_id}; "
                        "recompute because parameter version or batch size changed"
                    ),
                )
            )
            invalid_batches.append(batch.batch_id)
            for request_id in batch.request_ids:
                steps.append(
                    PropagationStep(
                        level="request_attribution",
                        worker_id=worker_id,
                        subject=request_id,
                        reason=(
                            f"request attributed to invalid batch {batch.batch_id}; "
                            "attribution and output summary must be recomputed"
                        ),
                    )
                )
                invalid_requests.add(request_id)

    return InvalidationReport(
        invalid_workers=invalid_workers,
        invalid_batch_ids=tuple(invalid_batches),
        invalid_request_ids=tuple(sorted(invalid_requests)),
        steps=tuple(steps),
        worker_reasons=worker_reasons,
    )
