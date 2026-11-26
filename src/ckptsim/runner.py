from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import CkptError
from .lease import Lease, LeaseScheduler, ScheduledTask
from .ledger import Ledger, SnapshotIdentity
from .model import (
    ParameterSet,
    build_batch_samples,
    compute_loss,
    generate_parameter_set,
)
from .plan import BatchPlan, EventPlan, build_event_plan
from .process import ShardProcessManager, check_ports_free
from .spec import SpecBundle


@dataclass
class RunnerOptions:
    ledger_path: str | Path
    lease_ttl_ms: int = 30_000
    client_timeout: float = 10.0
    startup_timeout: float = 8.0
    auto_recover: bool = True
    max_recovery_attempts: int = 2
    check_ports: bool = True


BatchStartedHook = Callable[[BatchPlan, ShardProcessManager], None]


def clean_runtime_artifacts(workdir: str | Path) -> list[str]:
    root = Path(workdir)
    removed: list[str] = []
    if not root.exists():
        return removed
    for path in sorted(root.iterdir()):
        if any(
            token in path.name
            for token in (".sqlite", ".snapshot.json", "ledger", "snapshot")
        ):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(str(path))
    return removed


class Runner:
    def __init__(
        self,
        bundle: SpecBundle,
        options: RunnerOptions,
        *,
        process_manager: ShardProcessManager | None = None,
        on_batch_started: BatchStartedHook | None = None,
    ) -> None:
        self.bundle = bundle
        self.options = options
        self.plan = build_event_plan(bundle)
        self._provided_manager = process_manager
        self.on_batch_started = on_batch_started
        self.parameter_cache: dict[str, dict[int, ParameterSet]] = {
            shard_id: {} for shard_id in bundle.shards.by_id()
        }

    def _parameters(self, shard_id: str, parameter_version: int) -> ParameterSet:
        shard = self.bundle.shards.by_id()[shard_id]
        version = self.bundle.layers.by_version()[parameter_version]
        return generate_parameter_set(
            shard, version, cache=self.parameter_cache[shard_id]
        )

    def _committed_for_plan(self, ledger: Ledger, batch: BatchPlan) -> dict[str, Any] | None:
        record = ledger.get_batch(batch.shard_id, batch.batch_id)
        if (
            record is not None
            and record["status"] == "committed"
            and record["parameter_version"] == batch.parameter_version
        ):
            return record
        return None

    def run(self) -> dict[str, Any]:
        ledger_path = Path(self.options.ledger_path)
        ledger = Ledger.open(
            ledger_path,
            identity=SnapshotIdentity.from_bundle(self.bundle),
            initial_versions=self.plan.initial_versions,
            initial_generation=0,
        )
        startup_recovered = ledger.recover_inflight("startup recovery from in-flight marker")
        manager = self._provided_manager
        owns_manager = manager is None
        if manager is None:
            if self.options.check_ports:
                check_ports_free(self.bundle.shards.shards)
            manager = ShardProcessManager(
                self.bundle.shards.shards,
                shards_path=self.bundle.paths["shards"],
                layers_path=self.bundle.paths["layers"],
                startup_timeout=self.options.startup_timeout,
                lease_ttl_ms=self.options.lease_ttl_ms,
                client_timeout=self.options.client_timeout,
            )
        scheduler = LeaseScheduler(
            global_lease_limit=self.bundle.shards.global_lease_limit,
            shard_lease_limits={
                shard.shard_id: shard.lease_limit for shard in self.bundle.shards.shards
            },
            shard_concurrency={
                shard.shard_id: shard.concurrency for shard in self.bundle.shards.shards
            },
            ttl_ms=self.options.lease_ttl_ms,
        )
        tasks: dict[tuple[str, str], ScheduledTask] = {}
        renewals: list[dict[str, Any]] = []
        recovery_events: list[dict[str, Any]] = list(startup_recovered)
        try:
            manager.start_all()
            pending = [
                batch
                for batch in self.plan.batches
                if self._committed_for_plan(ledger, batch) is None
            ]
            attempt = 0
            while pending:
                for batch in pending:
                    tasks[batch.key] = scheduler.submit(
                        seq=batch.event.seq,
                        shard_id=batch.shard_id,
                        task_id=batch.batch_id,
                        fn=lambda lease, batch=batch: self._execute_batch(
                            ledger, manager, batch, lease
                        ),
                    )
                failures: list[tuple[BatchPlan, BaseException]] = []
                for batch in pending:
                    try:
                        scheduler.wait(tasks[batch.key])
                    except BaseException as exc:  # inspect every task before deciding
                        failures.append((batch, exc))
                if not failures:
                    break
                marked = ledger.recover_inflight("batch failed before atomic commit")
                recovery_events.extend(marked)
                dead_shards = [
                    shard_id
                    for shard_id, process in manager.processes.items()
                    if process.process is None or process.process.poll() is not None
                ]
                attempt += 1
                if (
                    not self.options.auto_recover
                    or not dead_shards
                    or attempt > self.options.max_recovery_attempts
                ):
                    raise failures[0][1]
                for shard_id in dead_shards:
                    manager.restart(shard_id)
                    recovery_events.append(
                        {
                            "shard_id": shard_id,
                            "batch_id": "*",
                            "status": "worker_restarted",
                            "source": "committed_ledger",
                        }
                    )
                pending = [
                    batch
                    for batch in self.plan.batches
                    if self._committed_for_plan(ledger, batch) is None
                ]
            results = {
                batch.key: ledger.get_batch(batch.shard_id, batch.batch_id)
                for batch in self.plan.batches
            }
            if any(record is None for record in results.values()):
                missing = [key for key, record in results.items() if record is None]
                raise CkptError(
                    "E_VERIFY",
                    f"batches missing after execution: {missing}",
                    field="batches",
                )
            for renewal in self.plan.lease_renewals:
                key = (renewal["shard_id"], renewal["batch_id"])
                task = tasks.get(key)
                if task is None and ledger.lease_renewal_exists(
                    lease_id=renewal["lease_id"],
                    old_version=renewal["lease_version"],
                    new_version=renewal["lease_version"] + 1,
                ):
                    renewals.append(
                        {
                            **renewal,
                            "accepted": True,
                            "recovered_from_ledger": True,
                            "new_version": renewal["lease_version"] + 1,
                        }
                    )
                    continue
                if task is None:
                    raise CkptError(
                        "E_LEASE",
                        "cannot renew a batch lease that was not active in this session",
                        shard_id=key[0],
                        batch_id=key[1],
                    )
                client = manager.processes[key[0]].client
                assert client is not None
                response = scheduler.renew(
                    task,
                    expected_version=renewal["lease_version"],
                    validate=lambda lease, client=client, renewal=renewal: client.renew(
                        shard_id=renewal["shard_id"],
                        lease_id=renewal["lease_id"],
                        expected_version=renewal["lease_version"],
                        lease_ttl_ms=self.options.lease_ttl_ms,
                    ),
                )
                renewals.append(
                    {
                        **renewal,
                        "accepted": True,
                        "new_version": response["lease_version"],
                        "expires_ms": response["expires_ms"],
                    }
                )
                ledger.record_lease_renewal(
                    lease_id=renewal["lease_id"],
                    shard_id=renewal["shard_id"],
                    batch_id=renewal["batch_id"],
                    old_version=renewal["lease_version"],
                    new_version=response["lease_version"],
                    expires_ms=response["expires_ms"],
                )
            state = self._apply_state_events(ledger, results)
            batches = ledger.list_batches_by_seq()
            checkpoints = ledger.list_checkpoints()
            rollbacks = ledger.list_rollbacks()
            attempts = ledger.list_attempts()
            unknown_attempts = [item for item in attempts if item["status"] == "unknown"]
            snapshot_report = ledger.snapshot_report.to_dict() if ledger.snapshot_report else None
            return {
                "ok": True,
                "ledger": str(ledger_path),
                "snapshot": snapshot_report,
                "recovered_at_startup": startup_recovered,
                "recovery_events": recovery_events,
                "batches": batches,
                "checkpoints": checkpoints,
                "rollbacks": rollbacks,
                "renewals": renewals,
                "unknown_attempts": unknown_attempts,
                "scheduler": scheduler.snapshot(),
                "state": state,
                "forced_exit_report": None,
                "exit_report_note": (
                    "No exit report is produced for a forcibly terminated worker; "
                    "only ledger-committed state is authoritative."
                ),
            }
        finally:
            scheduler.close(wait=True)
            if owns_manager:
                manager.stop_all()
            ledger.close()

    def _execute_batch(
        self,
        ledger: Ledger,
        manager: ShardProcessManager,
        batch: BatchPlan,
        lease: Lease,
    ) -> dict[str, Any]:
        parameters = self._parameters(batch.shard_id, batch.parameter_version)
        shard = self.bundle.shards.by_id()[batch.shard_id]
        samples = build_batch_samples(
            shard, parameters, batch.batch_id, batch.sample_ids
        )
        attempt_id = ledger.begin_batch(
            batch,
            lease_id=batch.lease_id,
            lease_version=lease.version,
            granted_ms=lease.granted_ms,
            expires_ms=lease.expires_ms,
        )
        if self.on_batch_started is not None:
            self.on_batch_started(batch, manager)
        process = manager.processes[batch.shard_id]
        assert process.client is not None
        try:
            response = process.client.batch(
                shard_id=batch.shard_id,
                batch_id=batch.batch_id,
                sample_ids=list(batch.sample_ids),
                parameter_version=batch.parameter_version,
                samples=samples,
                lease_id=batch.lease_id,
                lease_version=lease.version,
                lease_ttl_ms=self.options.lease_ttl_ms,
            )
            if response["parameter_digest"] != parameters.digest:
                raise CkptError(
                    "E_VERIFY",
                    "worker parameter digest disagrees with independent deterministic generation",
                    shard_id=batch.shard_id,
                    batch_id=batch.batch_id,
                    field="parameter_digest",
                )
            expected_loss = compute_loss(
                shard,
                parameters,
                batch.batch_id,
                batch.sample_ids,
                samples=samples,
            )
            if response["loss_digest"] != expected_loss.digest:
                raise CkptError(
                    "E_VERIFY",
                    "worker loss digest disagrees with independent deterministic computation",
                    shard_id=batch.shard_id,
                    batch_id=batch.batch_id,
                    field="loss_digest",
                )
            checkpoint_status = "assigned" if batch.checkpoint_id else "unassigned"
            ledger.complete_batch(
                batch,
                response,
                checkpoint_id=batch.checkpoint_id,
                checkpoint_status=checkpoint_status,
                lease_id=batch.lease_id,
                lease_version=lease.version,
                attempt_id=attempt_id,
            )
            return ledger.get_batch(batch.shard_id, batch.batch_id) or {}
        except BaseException as exc:
            ledger.mark_batch_unknown(
                batch.shard_id,
                batch.batch_id,
                f"batch did not commit atomically: {type(exc).__name__}: {exc}",
            )
            raise

    def _apply_state_events(
        self,
        ledger: Ledger,
        results: dict[tuple[str, str], dict[str, Any] | None],
    ) -> dict[str, Any]:
        versions = dict(self.plan.initial_versions)
        generation = 0
        for event in self.bundle.events.events:
            if event.type == "checkpoint":
                plan = self.plan.checkpoints[event.data["checkpoint_id"]]
                baseline: list[dict[str, Any]] = []
                for shard_id, batch_id in sorted(plan.batch_keys):
                    record = results[(shard_id, batch_id)]
                    if record is None or record["status"] != "committed":
                        raise CkptError(
                            "E_VERIFY",
                            "checkpoint cannot freeze an uncommitted batch",
                            shard_id=shard_id,
                            batch_id=batch_id,
                            field="status",
                        )
                    baseline.append(
                        {
                            "shard_id": shard_id,
                            "batch_id": batch_id,
                            "parameter_version": record["parameter_version"],
                            "generation": record["generation"],
                            "sample_ids": record["sample_ids"],
                            "loss_digest": record["loss_digest"],
                        }
                    )
                parameter_digests = {
                    shard.shard_id: self._parameters(
                        shard.shard_id, plan.spec.parameter_version
                    ).digest
                    for shard in self.bundle.shards.shards
                }
                ledger.record_checkpoint(
                    plan,
                    parameter_digests=parameter_digests,
                    loss_baseline=baseline,
                )
            elif event.type == "rollback":
                rollback = next(
                    item
                    for item in self.plan.rollbacks
                    if item.event.seq == event.seq
                    and item.checkpoint_id == event.data["checkpoint_id"]
                )
                versions = {
                    shard_id: rollback.target_version for shard_id in versions
                }
                generation = rollback.generation_after
                ledger.record_rollback(rollback, current_versions=versions)
            elif event.type == "advance":
                parameter_version = event.data["parameter_version"]
                version = self.bundle.layers.by_version()[parameter_version]
                shard_ids = event.data.get("shard_ids", sorted(versions))
                for shard_id in shard_ids:
                    versions[shard_id] = parameter_version
                generation = max(generation, version.generation)
                ledger.record_advance(
                    seq=event.seq,
                    parameter_version=parameter_version,
                    generation=version.generation,
                    current_versions=versions,
                )
        return ledger.current_state()
