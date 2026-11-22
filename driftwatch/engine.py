from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .config import load_inputs
from .errors import DriftError
from .fixedpoint import judge_snapshot
from .ledger import Ledger
from .merge import merge_snapshot
from .models import (
    METRIC_INDEX,
    AgentSpec,
    CollectionTask,
    Criteria,
    DriftResult,
    MergedResult,
    Observation,
    Snapshot,
    SnapshotChain,
)
from .processes import AgentProcessManager
from .protocol import AgentClient
from .scheduler import ConcurrencyScheduler

LifecycleHook = Callable[[str, dict[str, Any]], None]


def observation_sequence(snapshot_seq: int, metric_id: str) -> int:
    return int(snapshot_seq) * 100 + METRIC_INDEX[metric_id]


def build_tasks(
    snapshot: Snapshot,
    agents: Iterable[AgentSpec],
) -> list[CollectionTask]:
    tasks: list[CollectionTask] = []
    for host_id in sorted(snapshot.hosts):
        for metric_id in snapshot.hosts[host_id].expected:
            sequence = observation_sequence(snapshot.seq, metric_id)
            for agent in sorted(agents, key=lambda item: item.agent_id):
                if agent.covers(host_id, metric_id):
                    tasks.append(
                        CollectionTask(
                            agent=agent,
                            snapshot_seq=snapshot.seq,
                            host_id=host_id,
                            metric_id=metric_id,
                            observation_seq=sequence,
                        )
                    )
    return sorted(tasks, key=lambda task: (task.agent.agent_id, task.host_id, task.metric_id))


def _collect_observations(
    tasks: list[CollectionTask],
    criteria: Criteria,
    batch_id: str,
) -> tuple[list[Observation], list[Any], dict[str, Any]]:
    scheduler = ConcurrencyScheduler(criteria)
    clients = {
        agent_id: AgentClient(agent, criteria.socket_timeout_ms, criteria.handshake_timeout_ms)
        for agent_id, agent in ((task.agent.agent_id, task.agent) for task in tasks)
    }
    outcome = scheduler.run(
        tasks,
        lambda task: clients[task.agent.agent_id].observe(task),
        batch_id=batch_id,
    )
    observations = sorted(
        outcome.successes.values(),
        key=lambda item: (item.snapshot_seq if hasattr(item, "snapshot_seq") else 0, item.host_id, item.metric_id, item.agent_id),
    )
    return observations, outcome.attempts, outcome.to_dict()


@dataclass
class EngineReport:
    generation: int
    replaced: bool
    affected: list[dict[str, Any]]
    differences: list[dict[str, Any]]
    recovered_unknown: list[dict[str, Any]]
    committed_batches: list[dict[str, Any]]
    pending_snapshots: list[int]
    replay: dict[str, Any] | None
    cursor: dict[str, Any]
    view_summary: dict[str, Any]
    reconcile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "replaced": self.replaced,
            "affected": self.affected,
            "differences": self.differences,
            "recovered_unknown": self.recovered_unknown,
            "committed_batches": self.committed_batches,
            "pending_snapshots": self.pending_snapshots,
            "replay": self.replay,
            "cursor": self.cursor,
            "view_summary": self.view_summary,
            "reconcile": self.reconcile,
        }


class DriftEngine:
    def __init__(
        self,
        chain: SnapshotChain,
        agents: tuple[AgentSpec, ...],
        criteria: Criteria,
        db_path: str,
        *,
        ledger: Ledger | None = None,
    ) -> None:
        self.chain = chain
        self.agents = tuple(sorted(agents, key=lambda item: item.agent_id))
        self.criteria = criteria
        self.db_path = db_path
        self.ledger = ledger or Ledger(db_path)

    @classmethod
    def from_files(
        cls,
        snapshot_path: str,
        agent_path: str,
        criteria_path: str | None,
        db_path: str,
    ) -> "DriftEngine":
        chain, agents, criteria = load_inputs(snapshot_path, agent_path, criteria_path)
        return cls(chain, agents, criteria, db_path)

    def close(self) -> None:
        self.ledger.close()

    def __enter__(self) -> "DriftEngine":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _view_summary(self, generation: int) -> dict[str, Any]:
        view = self.ledger.session_snapshot(generation)
        drift = view["drift_view"]
        return {
            "merged_rows": len(view["merged_view"]),
            "drift_rows": len(drift),
            "accepted": sum(1 for item in drift if item["decision"] == "ACCEPT"),
            "rejected": sum(1 for item in drift if item["decision"] == "REJECT"),
            "unknown": sum(1 for item in drift if item["decision"] == "UNKNOWN"),
            "sparse": sum(1 for item in view["merged_view"] if item["sparse"]),
            "error_affected": sum(1 for item in drift if item["error_affected"]),
            "conclusion_digest": view["conclusion_digest"],
        }

    def run(
        self,
        *,
        verify_replay: bool = False,
        cleanup_stale: bool = True,
        lifecycle_hook: LifecycleHook | None = None,
    ) -> EngineReport:
        reconcile = self.ledger.reconcile_session(self.chain, self.agents, self.criteria)
        generation = reconcile.generation
        recovered = self.ledger.recover_incomplete()
        current_recovered = [item for item in recovered if item["generation"] == generation]
        known_recovery_keys = {(item["snapshot_seq"], item["host_id"], item["metric_id"]) for item in current_recovered}
        for item in self.ledger.required_recollection(generation):
            key = (item["snapshot_seq"], item["host_id"], item["metric_id"])
            if key not in known_recovery_keys:
                current_recovered.append(
                    {
                        "generation": item["generation"],
                        "batch_id": item["batch_id"],
                        "snapshot_seq": item["snapshot_seq"],
                        "host_id": item["host_id"],
                        "metric_id": item["metric_id"],
                        "required_recollect": True,
                    }
                )
        current_recovered.sort(key=lambda item: (item["snapshot_seq"], item["host_id"], item["metric_id"]))
        committed_seqs = set(self.ledger.committed_snapshot_seqs(generation))
        pending = [snapshot for snapshot in self.chain.ordered() if snapshot.seq not in committed_seqs]
        replay_report: dict[str, Any] | None = None
        manager: AgentProcessManager | None = None
        committed_now: list[dict[str, Any]] = []
        try:
            if verify_replay and committed_seqs:
                manager = AgentProcessManager(self.agents, self.criteria.handshake_timeout_ms)
                manager.start(cleanup_stale=cleanup_stale)
                replay_report = self._replay_committed(generation)
            if pending:
                if manager is None:
                    manager = AgentProcessManager(self.agents, self.criteria.handshake_timeout_ms)
                    manager.start(cleanup_stale=cleanup_stale)
                for snapshot in pending:
                    committed_now.append(self._collect_and_commit(generation, snapshot, lifecycle_hook))
        finally:
            if manager is not None:
                manager.stop()
        return EngineReport(
            generation=generation,
            replaced=reconcile.replaced,
            affected=[item.to_dict() for item in reconcile.affected],
            differences=reconcile.differences,
            recovered_unknown=current_recovered,
            committed_batches=committed_now,
            pending_snapshots=[snapshot.seq for snapshot in pending],
            replay=replay_report,
            cursor=self.ledger.cursor(generation),
            view_summary=self._view_summary(generation),
            reconcile=reconcile.to_dict(),
        )

    def _collect_and_commit(
        self,
        generation: int,
        snapshot: Snapshot,
        lifecycle_hook: LifecycleHook | None,
    ) -> dict[str, Any]:
        tasks = build_tasks(snapshot, self.agents)
        planned = [
            {"snapshot_seq": snapshot.seq, "host_id": host_id, "metric_id": metric_id}
            for host_id in sorted(snapshot.hosts)
            for metric_id in sorted(snapshot.hosts[host_id].expected, key=METRIC_INDEX.__getitem__)
        ]
        batch_id = self.ledger.begin_batch(generation, snapshot.seq, planned)
        if lifecycle_hook is not None:
            lifecycle_hook(
                "batch_begun",
                {"batch_id": batch_id, "generation": generation, "snapshot_seq": snapshot.seq},
            )
        try:
            observations, attempts, scheduler_stats = _collect_observations(tasks, self.criteria, batch_id)
            merged = merge_snapshot(snapshot, observations, self.criteria.min_sources)
            drift = judge_snapshot(merged, snapshot.hosts, self.criteria)
            cursor = {
                "generation": generation,
                "snapshot_seq": snapshot.seq,
                "host_metric_count": len(planned),
                "observation_count": len(observations),
                "attempt_count": len(attempts),
                "scheduler": scheduler_stats,
            }
            commit = self.ledger.commit_batch(
                batch_id,
                generation=generation,
                snapshot_seq=snapshot.seq,
                observations=observations,
                merged_results=merged,
                drift_results=drift,
                attempts=attempts,
                cursor=cursor,
            )
            if lifecycle_hook is not None:
                lifecycle_hook(
                    "batch_committed",
                    {"batch_id": batch_id, "generation": generation, "snapshot_seq": snapshot.seq},
                )
            return commit
        except BaseException:
            try:
                self.ledger.abort_batch(batch_id, "collection or commit failed")
            except DriftError:
                pass
            raise

    def _replay_committed(self, generation: int) -> dict[str, Any]:
        batches = self.ledger.committed_batches(generation)
        matched = 0
        for batch in batches:
            batch_id = batch["batch_id"]
            snapshot_seq = int(batch["snapshot_seq"])
            snapshot = self.chain.snapshots.get(snapshot_seq)
            if snapshot is None:
                raise DriftError(
                    "E_REPLAY",
                    "ledger references a snapshot absent from the declared chain",
                    details={"snapshot_seq": snapshot_seq, "field": "snapshot_chain", "batch_id": batch_id},
                )
            tasks = build_tasks(snapshot, self.agents)
            observations, _, _ = _collect_observations(tasks, self.criteria, batch_id)
            actual_merged = merge_snapshot(snapshot, observations, self.criteria.min_sources)
            actual_drift = judge_snapshot(actual_merged, snapshot.hosts, self.criteria)
            reference = self.ledger.batch_results(batch_id)
            matched += self._compare_results(
                batch_id=batch_id,
                snapshot_seq=snapshot_seq,
                actual_merged=actual_merged,
                actual_drift=actual_drift,
                reference=reference,
            )
        return {
            "ok": True,
            "generation": generation,
            "batch_count": len(batches),
            "matched_rows": matched,
            "monotonic_fields_excluded": True,
        }

    def _compare_results(
        self,
        *,
        batch_id: str,
        snapshot_seq: int,
        actual_merged: list[MergedResult],
        actual_drift: list[DriftResult],
        reference: dict[str, list[dict[str, Any]]],
    ) -> int:
        expected_merged = {(row["host_id"], row["metric_id"]): row for row in reference["merged"]}
        expected_drift = {(row["host_id"], row["metric_id"]): row for row in reference["drift"]}
        actual_merged_map = {(row.host_id, row.metric_id): row for row in actual_merged}
        actual_drift_map = {(row.host_id, row.metric_id): row for row in actual_drift}
        merged_fields = (
            "status",
            "authority_agent",
            "authority_observation_seq",
            "authority_value",
            "aggregation_version",
            "source_count",
            "sparse",
            "max_abs_deviation",
            "error_code",
            "sources",
        )
        drift_fields = (
            "status",
            "decision",
            "expected",
            "observed",
            "tolerance_bps",
            "drift_signed_fixed",
            "drift_excess_fixed",
            "drift_ratio_bps",
            "allowed_fixed",
            "error_bound",
            "error_affected",
            "boundary_rule",
            "saturated",
            "error_code",
            "reason",
        )
        all_keys = sorted(set(expected_merged) | set(actual_merged_map) | set(expected_drift) | set(actual_drift_map))
        matched = 0
        for key in all_keys:
            expected_merged_row = expected_merged.get(key)
            actual_merged_row = actual_merged_map.get(key)
            if expected_merged_row is None or actual_merged_row is None:
                raise self._replay_mismatch(
                    snapshot_seq,
                    key[0],
                    key[1],
                    "merged.presence",
                    expected_merged_row is not None,
                    actual_merged_row is not None,
                    batch_id,
                )
            expected_merged_value = dict(expected_merged_row)
            expected_merged_value["sparse"] = bool(expected_merged_value["sparse"])
            expected_sources = self._strip_monotonic(expected_merged_value["sources"])
            actual_sources = self._strip_monotonic(actual_merged_row.to_dict()["sources"])
            expected_merged_value["sources"] = expected_sources
            actual_merged_value = actual_merged_row.to_dict()
            actual_merged_value["sources"] = actual_sources
            for field in merged_fields:
                if expected_merged_value[field] != actual_merged_value[field]:
                    raise self._replay_mismatch(
                        snapshot_seq,
                        key[0],
                        key[1],
                        f"merged.{field}",
                        expected_merged_value[field],
                        actual_merged_value[field],
                        batch_id,
                    )
            matched += 1

            expected_drift_row = expected_drift.get(key)
            actual_drift_row = actual_drift_map.get(key)
            if expected_drift_row is None or actual_drift_row is None:
                raise self._replay_mismatch(
                    snapshot_seq,
                    key[0],
                    key[1],
                    "drift.presence",
                    expected_drift_row is not None,
                    actual_drift_row is not None,
                    batch_id,
                )
            expected_drift_value = {
                "status": expected_drift_row["status"],
                "decision": expected_drift_row["decision"],
                "expected": expected_drift_row["expected"],
                "observed": expected_drift_row["observed"],
                "tolerance_bps": expected_drift_row["tolerance_bps"],
                "drift_signed_fixed": expected_drift_row["drift_signed_fixed"],
                "drift_excess_fixed": expected_drift_row["drift_excess_fixed"],
                "drift_ratio_bps": expected_drift_row["drift_ratio_bps"],
                "allowed_fixed": expected_drift_row["allowed_fixed"],
                "error_bound": expected_drift_row["error_bound"],
                "error_affected": bool(expected_drift_row["error_affected"]),
                "boundary_rule": expected_drift_row["boundary_rule"],
                "saturated": bool(expected_drift_row["saturated"]),
                "error_code": expected_drift_row["error_code"],
                "reason": expected_drift_row["reason"],
            }
            actual_drift_value = actual_drift_row.to_dict()
            for field in drift_fields:
                if expected_drift_value[field] != actual_drift_value[field]:
                    raise self._replay_mismatch(
                        snapshot_seq,
                        key[0],
                        key[1],
                        f"drift.{field}",
                        expected_drift_value[field],
                        actual_drift_value[field],
                        batch_id,
                    )
        actual_only = sorted((set(actual_merged_map) | set(actual_drift_map)) - (set(expected_merged) | set(expected_drift)))
        if actual_only:
            key = actual_only[0]
            raise self._replay_mismatch(snapshot_seq, key[0], key[1], "row.presence", False, True, batch_id)
        return matched

    @staticmethod
    def _strip_monotonic(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in source.items() if key != "monotonic_ms"}
            for source in sources
        ]

    @staticmethod
    def _replay_mismatch(
        snapshot_seq: int,
        host_id: str,
        metric_id: str,
        field: str,
        expected: Any,
        actual: Any,
        batch_id: str,
    ) -> DriftError:
        return DriftError(
            "E_REPLAY",
            "replayed conclusion differs from the committed ledger",
            details={
                "snapshot_seq": snapshot_seq,
                "host_id": host_id,
                "metric_id": metric_id,
                "field": field,
                "expected": expected,
                "actual": actual,
                "batch_id": batch_id,
            },
        )


def replay_ledger(db_path: str, *, cleanup_stale: bool = True) -> dict[str, Any]:
    ledger = Ledger(db_path)
    try:
        recovered = ledger.recover_incomplete()
        generation = ledger.active_generation()
        if generation is None:
            raise DriftError("E_REPLAY", "ledger has no active generation")
        chain, agents, criteria = ledger.generation_declarations(generation)
        engine = DriftEngine(chain, agents, criteria, db_path, ledger=ledger)
        manager = AgentProcessManager(agents, criteria.handshake_timeout_ms)
        manager.start(cleanup_stale=cleanup_stale)
        try:
            report = engine._replay_committed(generation)
        finally:
            manager.stop()
        return {
            "ok": True,
            "generation": generation,
            "recovered_unknown": [item for item in recovered if item["generation"] == generation],
            "required_recollections": ledger.required_recollection(generation),
            "replay": report,
            "cursor": ledger.cursor(generation),
        }
    finally:
        ledger.close()


def recover_ledger(db_path: str, *, cleanup_stale: bool = True) -> dict[str, Any]:
    ledger = Ledger(db_path)
    try:
        generation = ledger.active_generation()
        if generation is None:
            raise DriftError("E_REPLAY", "ledger has no active generation")
        chain, agents, criteria = ledger.generation_declarations(generation)
        engine = DriftEngine(chain, agents, criteria, db_path, ledger=ledger)
        report = engine.run(verify_replay=True, cleanup_stale=cleanup_stale)
        return {
            "ok": True,
            "generation": generation,
            "source": "ledger_declarations",
            "report": report.to_dict(),
            "required_recollections": ledger.required_recollection(generation),
        }
    finally:
        ledger.close()
