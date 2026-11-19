"""End-to-end evolution session orchestration."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.compiler import CompiledVersion, compile_version
from ..config.grid import Grid
from ..config.incremental import incremental_compile
from ..config.manifest import Manifest, ManifestVersion, Registry
from ..config.planner import DistributionPlan, build_plan
from ..persistence.database import Database
from ..persistence.snapshot import SnapshotIdentity, SnapshotStore
from ..util.clock import now_ms
from ..util.logging_util import StructuredLogger
from .converge import ConvergenceReport, evaluate_convergence
from .scheduler import NodePool, Scheduler, SchedulerResult
from .state_machine import StateMachine
from .view import build_view, mark_stale, view_payload


@dataclass
class VersionRunResult:
    version: int
    state: str
    invalidated: set[str]
    invalidation_steps: list[dict[str, str]]
    byte_check: dict[str, bool]
    plan: DistributionPlan
    scheduler: SchedulerResult
    convergence: ConvergenceReport | None
    compiled: CompiledVersion
    view: dict[str, Any]
    identity: SnapshotIdentity


class Session:
    def __init__(
        self,
        session_id: str,
        manifest: Manifest,
        registry: Registry,
        grid: Grid,
        db: Database,
        snapshot_store: SnapshotStore,
        logger: StructuredLogger,
        *,
        concurrency: int = 2,
        max_attempts: int = 3,
        log_dir: str = "logs",
        inject_by_node: dict[str, list[str]] | None = None,
        startup_timeout_ms: int = 8000,
        handshake_timeout_ms: int = 5000,
        operation_timeout_ms: int = 10000,
    ) -> None:
        self.session_id = session_id
        self.manifest = manifest
        self.registry = registry
        self.grid = grid
        self.db = db
        self.snapshot_store = snapshot_store
        self.logger = logger
        self.concurrency = concurrency
        self.max_attempts = max_attempts
        self.log_dir = log_dir
        self.inject_by_node = inject_by_node or {}
        self.state_machine = StateMachine()
        self.created_at_ms = now_ms()
        self._previous_manifest: ManifestVersion | None = None
        self._previous_compiled: CompiledVersion | None = None
        self._db_registered = False

    def _register(self) -> None:
        if self._db_registered:
            return
        self.db.upsert_session(
            self.session_id,
            self.created_at_ms,
            self.grid.q,
            self.state_machine.state,
            self.state_machine.history,
        )
        self._db_registered = True

    def run_version(self, version: ManifestVersion) -> VersionRunResult:
        self._register()
        self.logger.log(
            "version_start",
            source="session",
            key=str(version.version),
            version=version.version,
        )

        compiled, invalidated, steps, byte_check = incremental_compile(
            self._previous_manifest,
            version,
            self.grid,
            self._previous_compiled,
        )
        self.state_machine.transition("COMPILED")
        self.db.upsert_session(
            self.session_id,
            self.created_at_ms,
            self.grid.q,
            self.state_machine.state,
            self.state_machine.history,
        )
        self.logger.log(
            "version_compiled",
            source="compiler",
            key=str(version.version),
            invalidated=sorted(invalidated),
            digest=compiled.content_digest,
        )

        plan = build_plan(version, compiled, self.grid.q)
        self.state_machine.transition("DISPATCHING")

        pool = NodePool(
            self.registry,
            self.logger,
            log_dir=self.log_dir,
            inject_by_node=self.inject_by_node,
            startup_timeout_ms=8000,
            handshake_timeout_ms=5000,
            operation_timeout_ms=10000,
        )
        convergence: ConvergenceReport | None = None
        try:
            node_ids = sorted({request.node_id for request in plan.requests})
            pool.start(node_ids)
            scheduler = Scheduler(pool, self.logger, concurrency=self.concurrency, max_attempts=self.max_attempts)
            scheduler_result = scheduler.run(plan)
        finally:
            pool.shutdown_all()

        try:
            convergence = evaluate_convergence(
                plan, version, compiled, self.registry, scheduler_result
            )
            self.state_machine.transition("CONVERGED")
        except Exception:
            self.state_machine.transition("DIVERGED")
            raise
        finally:
            rows = build_view(version, compiled, convergence) if convergence else []
            rows = mark_stale(rows, version.version)
            view = view_payload(rows)
            payload = self._build_payload(
                version, compiled, plan, scheduler_result, convergence, steps, view
            )
            identity = self.snapshot_store.save(self.session_id, version.version, payload)
            self.state_machine.transition("PERSISTED")
            self.db.upsert_session(
                self.session_id,
                self.created_at_ms,
                self.grid.q,
                self.state_machine.state,
                self.state_machine.history,
            )

        result = VersionRunResult(
            version=version.version,
            state=self.state_machine.state,
            invalidated=invalidated,
            invalidation_steps=[
                {"key": s.key, "trigger": s.trigger, "source": s.source} for s in steps
            ],
            byte_check=byte_check,
            plan=plan,
            scheduler=scheduler_result,
            convergence=convergence,
            compiled=compiled,
            view=view,
            identity=identity,
        )
        self._previous_manifest = version
        self._previous_compiled = compiled
        return result

    def _build_payload(
        self,
        version: ManifestVersion,
        compiled: CompiledVersion,
        plan: DistributionPlan,
        scheduler_result: SchedulerResult,
        convergence: ConvergenceReport | None,
        steps: list[Any],
        view: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "version": version.version,
            "grid": self.grid.to_dict(),
            "manifest": version.to_dict(),
            "plan": plan.to_dict(),
            "errors": {key: compiled.errors[key].to_dict() for key in compiled.order},
            "artifacts": {
                key: {
                    "artifact_id": compiled.items[key].artifact_id,
                    "artifact_hex": compiled.items[key].artifact_bytes.hex(),
                    "error": compiled.items[key].error.to_dict(),
                }
                for key in compiled.order
            },
            "attempts": [
                attempt.to_dict()
                for record in scheduler_result.records
                for attempt in record.attempts
            ],
            "requests": [record.to_dict() for record in scheduler_result.records],
            "receipts": scheduler_result.receipts,
            "convergence": convergence.to_dict() if convergence else None,
            "invalidation_steps": [
                {"key": s.key, "trigger": s.trigger, "source": s.source} for s in steps
            ],
            "view": view,
        }


def new_session_id() -> str:
    return "ses-" + uuid.uuid4().hex[:12]
