"""Crash recovery, deterministic replay and snapshot restore services."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..errors import ReplayError, SnapshotError
from ..models import Decision, Node, Pod
from ..scheduler.engine import SchedulingEngine, SessionParams
from ..scheduler.filter import Placement
from .ledger import STATUS_COMMITTED, STATUS_UNKNOWN, Ledger


@dataclass
class RecoveryResult:
    restored_pods: List[str] = field(default_factory=list)
    unknown_pods: List[str] = field(default_factory=list)
    parameter_version: Optional[int] = None
    replay_errors: List[str] = field(default_factory=list)
    verified: bool = False

    def summary(self) -> Dict[str, Any]:
        return {
            "restored_pods": list(self.restored_pods),
            "unknown_pods": list(self.unknown_pods),
            "parameter_version": self.parameter_version,
            "replay_errors": list(self.replay_errors),
            "verified": self.verified,
        }


def _compare_values(pod_id: str, field_name: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        raise ReplayError(
            f"replay mismatch for {field_name}: ledger={expected!r} replay={actual!r}",
            pod_id=pod_id,
            field=field_name,
        )


def _compare_decision(pod_id: str, stored: Dict[str, Any], replayed: Decision) -> None:
    """Field-by-field comparison; the first mismatch raises E_REPLAY."""
    _compare_values(pod_id, "feasible", stored["feasible"], replayed.feasible)
    _compare_values(pod_id, "candidates", list(stored["candidates"]),
                    list(replayed.candidates))
    stored_rejections = [
        (item["node"], item["gate"], item["reason"]) for item in stored["rejections"]
    ]
    replay_rejections = [
        (item.node, item.gate, item.reason) for item in replayed.rejections
    ]
    _compare_values(pod_id, "rejections", stored_rejections, replay_rejections)
    _compare_values(pod_id, "bound_node", stored["bound_node"], replayed.bound_node)
    _compare_values(pod_id, "tie_reason", stored["tie_reason"], replayed.tie_reason)
    _compare_values(
        pod_id, "error_influenced",
        stored["error_influenced"], replayed.error_influenced,
    )
    _compare_values(pod_id, "score_gap_mp", stored["score_gap_mp"],
                    replayed.score_gap_mp)
    _compare_values(pod_id, "combined_error_mp", stored["combined_error_mp"],
                    replayed.combined_error_mp)

    stored_scores = {item["node"]: item for item in stored["node_scores"]}
    replay_scores = {item.node: item for item in replayed.node_scores}
    _compare_values(pod_id, "scored_nodes",
                    sorted(stored_scores), sorted(replay_scores))
    for node_name in sorted(stored_scores):
        left = stored_scores[node_name]
        right = replay_scores[node_name]
        prefix = f"node_scores[{node_name}]"
        _compare_values(pod_id, f"{prefix}.total_mp",
                        left["total_mp"], right.total_mp)
        _compare_values(pod_id, f"{prefix}.total_points",
                        left["total_points"], right.total_points)
        _compare_values(pod_id, f"{prefix}.error_bound_mp",
                        left["error_bound_mp"], right.error_bound_mp)
        _compare_values(pod_id, f"{prefix}.error_bound_points",
                        left["error_bound_points"], right.error_bound_points)
        left_plugins = {p["plugin"]: p for p in left["plugins"]}
        right_plugins = {p.plugin: p for p in right.plugins}
        _compare_values(pod_id, f"{prefix}.plugin_ids",
                        sorted(left_plugins), sorted(right_plugins))
        for plugin in sorted(left_plugins):
            lp = left_plugins[plugin]
            rp = right_plugins[plugin]
            _compare_values(
                pod_id, f"{prefix}.plugins[{plugin}].raw_score",
                lp["raw_score"], rp.raw_score,
            )
            _compare_values(
                pod_id, f"{prefix}.plugins[{plugin}].contribution_mp",
                lp["contribution_mp"], rp.contribution_mp,
            )
            _compare_values(
                pod_id, f"{prefix}.plugins[{plugin}].quant_error_mp",
                lp["quant_error_mp"], rp.quant_error_mp,
            )


def recover_and_replay(
    ledger: Ledger,
    engine: SchedulingEngine,
    stop_at_unknown: bool = True,
) -> RecoveryResult:
    """Restore committed pods and replay Filter+Score under the stored version."""
    result = RecoveryResult()
    rows = ledger.load_pod_results()
    if not rows:
        result.verified = True
        return result

    params = ledger.latest_parameter_version()
    if params is not None:
        engine.params = params
        result.parameter_version = params.version

    capacities = engine.fresh_capacities()
    placements: List[Placement] = []
    pod_lookup = {pod.pod_id: pod for pod in engine.pods}

    for row in rows:
        pod_id = row["pod_id"]
        if row["status"] == STATUS_UNKNOWN:
            result.unknown_pods.append(pod_id)
            if stop_at_unknown:
                break
            continue
        pod = pod_lookup[pod_id]
        replayed = engine.schedule_one(pod, capacities, placements)
        stored = row["decision"]
        _compare_decision(pod_id, stored, replayed)
        result.restored_pods.append(pod_id)
        if replayed.bound_node is not None:
            bound = replayed.bound_node
            capacities[bound].cpu_used += pod.cpu_request
            capacities[bound].memory_used += pod.memory_request
            placements.append(
                Placement(pod_id=pod.pod_id, labels=dict(pod.labels), node=bound)
            )

    result.verified = not result.unknown_pods
    return result


def rerun_unknown_pods(
    ledger: Ledger,
    engine: SchedulingEngine,
) -> List[Decision]:
    """Recompute from the first UNKNOWN pod and append fresh committed records."""
    rows = ledger.load_pod_results()
    capacities = engine.fresh_capacities()
    placements: List[Placement] = []
    pod_lookup = {pod.pod_id: pod for pod in engine.pods}
    decisions: List[Decision] = []
    unknown_started = False

    for row in rows:
        pod = pod_lookup[row["pod_id"]]
        if row["status"] == STATUS_COMMITTED and not unknown_started:
            decision = engine.schedule_one(pod, capacities, placements)
            if decision.bound_node is not None:
                capacities[decision.bound_node].cpu_used += pod.cpu_request
                capacities[decision.bound_node].memory_used += pod.memory_request
                placements.append(
                    Placement(pod_id=pod.pod_id, labels=dict(pod.labels),
                              node=decision.bound_node)
                )
            continue
        unknown_started = True
        decision = engine.schedule_one(pod, capacities, placements)
        ledger.commit_decision(row["seq"], pod.pod_id, decision)
        decisions.append(decision)
        if decision.bound_node is not None:
            capacities[decision.bound_node].cpu_used += pod.cpu_request
            capacities[decision.bound_node].memory_used += pod.memory_request
            placements.append(
                Placement(pod_id=pod.pod_id, labels=dict(pod.labels),
                          node=decision.bound_node)
            )
    return decisions


def restore_snapshot_or_replace(
    ledger: Ledger,
    snapshot_id: str,
    nodes: Sequence[Node],
    pods: Sequence[Pod],
    params: SessionParams,
    current_spec_digest: str,
) -> RecoveryResult:
    """Verify a snapshot; on mismatch, replace the whole ledger state."""
    try:
        ledger.verify_snapshot(snapshot_id, current_spec_digest, params)
    except SnapshotError as exc:
        # Whole-state replacement: never incrementally merge stale content.
        ledger.reset_pod_results()
        ledger.replace_specs(nodes, pods)
        ledger.record_parameter_version(params)
        result = RecoveryResult(parameter_version=params.version)
        result.replay_errors = list(exc.mismatches)
        return result

    engine = SchedulingEngine(nodes=list(nodes), pods=list(pods), params=params)
    return recover_and_replay(ledger, engine)
