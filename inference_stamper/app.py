"""演示与命令行共用的编排逻辑。"""

from __future__ import annotations
from pathlib import Path

from .ledger import LedgerSession, open_ledger
from .parameters import IdentityArtifact, build_identity
from .recovery import recover, stamp_batch_with_crash_point
from .replay import replay_ledger
from .scheduler import Task, TraceEntry, deterministic_duration, plan_virtual_schedule, run_threaded
from .spec import Spec
from .stamping import Stamper, StampedToken, VerifyResult


def build_identities(spec: Spec) -> dict[str, IdentityArtifact]:
    return {
        backend.backend_id: build_identity(
            backend_id=backend.backend_id,
            model_dim=backend.model_dim,
            hidden_dim=backend.hidden_dim,
            seed=backend.seed,
            quant=backend.quant,
            identity_version=backend.identity_version,
            token_spec_version=backend.token_spec_version,
        )
        for backend in spec.backends
    }


def stamp_all(
    stamper: Stamper,
    spec: Spec,
    crash_after_token_id: str | None = None,
) -> list[StampedToken]:
    return stamp_batch_with_crash_point(
        stamper, list(spec.requests), crash_after_token_id)


def verify_all(stamper: Stamper, spec: Spec) -> list[VerifyResult]:
    return stamper.verify_all(spec)


def build_scheduler_tasks(spec: Spec, master_seed: int) -> list[Task]:
    tasks: list[Task] = []
    for order, request in enumerate(spec.requests):
        tasks.append(Task(
            task_id=request.token_id,
            backend_id=request.backend_id,
            kind="stamp",
            duration_ms=deterministic_duration(
                request.token_id, request.backend_id, "stamp",
                master_seed),
            order=order,
        ))
    for offset, request in enumerate(spec.requests):
        tasks.append(Task(
            task_id=request.token_id + "-verify",
            backend_id=request.backend_id,
            kind="verify",
            duration_ms=deterministic_duration(
                request.token_id, request.backend_id, "verify",
                master_seed),
            order=len(spec.requests) + offset,
        ))
    return tasks


def run_virtual_demo(spec: Spec, master_seed: int,
                     global_limit: int,
                     per_backend_limit: int) -> list[TraceEntry]:
    tasks = build_scheduler_tasks(spec, master_seed)
    return plan_virtual_schedule(tasks, global_limit, per_backend_limit)


def run_threaded_demo(
    stamper: Stamper,
    spec: Spec,
    master_seed: int,
    global_limit: int,
    per_backend_limit: int,
) -> list[TraceEntry]:
    tasks = build_scheduler_tasks(spec, master_seed)

    def worker(task: Task) -> None:
        if task.kind == "stamp":
            request = next(r for r in spec.requests
                           if r.token_id == task.task_id)
            stamper.stamp(request)
        else:
            token_id = task.task_id.removesuffix("-verify")
            if stamper.ledger.get_stamp(token_id) is not None:
                stamper.verify(token_id)

    return run_threaded(tasks, worker, global_limit, per_backend_limit)


def open_run_ledger(path: str | Path, *, on_mismatch: str = "replace"):
    return open_ledger(path, on_mismatch=on_mismatch)


def recover_and_replay(ledger: LedgerSession, spec: Spec):
    report = recover(ledger, spec)
    replay_items = replay_ledger(ledger, spec)
    return report, replay_items
