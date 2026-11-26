"""High-level cluster orchestration for real child nodes and incremental reruns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .client import NodeProcess, execute_shard
from .invalidation import InvalidationPlan
from .ledger import Ledger, verify_preserved_bytes
from .model import ShardSpec, Spec
from .samples import canonical_json_bytes, generate_samples
from .scheduler import LeaseHandle, RunResult, Scheduler


@dataclass
class IncrementalResult:
    records: dict[str, dict[str, Any]]
    run: RunResult
    preserved: dict[str, Any]
    recomputed_shards: tuple[str, ...]

    def ordered_records(self, spec: Spec) -> list[dict[str, Any]]:
        return [self.records[shard_id] for shard_id in spec.topological_order() if shard_id in self.records]


class ClusterRuntime:
    def __init__(
        self,
        spec: Spec,
        *,
        ledger: Ledger | None = None,
        session_version: int = 1,
        batch_size_overrides: dict[str, int] | None = None,
    ) -> None:
        self.spec = spec
        self.ledger = ledger
        self.session_version = session_version
        self.batch_size_overrides = dict(batch_size_overrides or {})
        self.nodes = spec.node_map()
        self.processes: dict[str, NodeProcess] = {}
        self.samples: dict[str, list[dict[str, Any]]] = {
            shard.id: generate_samples(self.nodes[shard.node_id], shard) for shard in spec.shards
        }

    def start(self) -> None:
        if self.processes:
            return
        try:
            for node in self.spec.nodes:
                process = NodeProcess(
                    node,
                    startup_timeout_s=max(0.1, self.spec.handshake_timeout_ms / 1000),
                )
                process.start()
                self.processes[node.id] = process
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        for process in reversed(list(self.processes.values())):
            process.stop()
        self.processes.clear()

    def __enter__(self) -> "ClusterRuntime":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def _task(self, shard: ShardSpec, lease: LeaseHandle) -> dict[str, Any]:
        node = self.nodes[shard.node_id]
        return execute_shard(
            node,
            shard,
            self.samples[shard.id],
            self.spec.capacity_version,
            read_timeout_s=self.spec.io_timeout_ms / 1000,
            handshake_timeout_s=self.spec.handshake_timeout_ms / 1000,
            batch_size=self.batch_size_overrides.get(shard.id),
            before_batch=lease.renew,
        )

    def execute(
        self,
        *,
        selected_shards: Iterable[str] | None = None,
        prior_done: Iterable[str] = (),
        prior_records: dict[str, dict[str, Any]] | None = None,
    ) -> RunResult:
        self.start()
        scheduler = Scheduler(
            self.spec,
            task=self._task,
            session_version=self.session_version,
            ledger=self.ledger,
        )
        return scheduler.run(
            selected_shards=selected_shards,
            prior_done=prior_done,
            prior_records=prior_records,
        )

    def recompute_subset(
        self,
        previous_records: dict[str, dict[str, Any]],
        plan: InvalidationPlan,
    ) -> IncrementalResult:
        if not plan.invalid_shards:
            empty = RunResult(
                records={},
                waits=[],
                lease_events=[],
                session_version=self.session_version,
            )
            return IncrementalResult(
                records={key: previous_records[key] for key in self.spec.topological_order() if key in previous_records},
                run=empty,
                preserved={"preserved": True, "shards_checked": list(plan.preserved_shards)},
                recomputed_shards=(),
            )
        preserved_records = {
            shard_id: previous_records[shard_id]
            for shard_id in plan.preserved_shards
        }
        if len(preserved_records) != len(plan.preserved_shards):
            missing = sorted(set(plan.preserved_shards) - set(preserved_records))
            raise KeyError(f"previous results missing preserved shards: {missing}")
        if self.ledger is not None:
            self.ledger.import_records(preserved_records.values())
        run = self.execute(
            selected_shards=plan.invalid_shards,
            prior_done=plan.preserved_shards,
            prior_records={},
        )
        merged: dict[str, dict[str, Any]] = {}
        for shard_id in self.spec.topological_order():
            if shard_id in run.records:
                merged[shard_id] = run.records[shard_id]
            elif shard_id in preserved_records:
                merged[shard_id] = preserved_records[shard_id]
        preserved = verify_preserved_bytes(previous_records, merged, plan.preserved_shards)
        return IncrementalResult(
            records=merged,
            run=run,
            preserved=preserved,
            recomputed_shards=plan.invalid_shards,
        )


def records_canonical_bytes(records: dict[str, dict[str, Any]], spec: Spec) -> dict[str, bytes]:
    return {
        shard_id: canonical_json_bytes(records[shard_id])
        for shard_id in spec.topological_order()
        if shard_id in records
    }
