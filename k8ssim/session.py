"""High-level session facade: engine + ledger, designed for UI and demo use."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .loader import load_nodes, load_pods
from .models import Decision, Node, Pod
from .scheduler.engine import (
    ScheduleReport,
    SchedulingEngine,
    SessionParams,
    default_params,
)
from .scheduler.filter import Placement
from .storage.ledger import Ledger
from .storage.recovery import RecoveryResult, recover_and_replay, rerun_unknown_pods


@dataclass
class Session:
    nodes: List[Node]
    pods: List[Pod]
    ledger: Ledger
    engine: SchedulingEngine

    @classmethod
    def create(
        cls,
        nodes_path: str | Path,
        pods_path: str | Path,
        ledger_path: str | Path,
        replace: bool = True,
    ) -> "Session":
        nodes = load_nodes(nodes_path)
        pods = load_pods(pods_path, nodes)
        ledger = Ledger(ledger_path)
        if replace:
            ledger.reset_pod_results()
        ledger.replace_specs(nodes, pods)
        params = default_params()
        ledger.record_parameter_version(params)
        engine = SchedulingEngine(nodes=nodes, pods=pods, params=params)
        return cls(nodes=nodes, pods=pods, ledger=ledger, engine=engine)

    @classmethod
    def open(cls, ledger_path: str | Path) -> "Session":
        ledger = Ledger(ledger_path)
        nodes, pods, _ = ledger.load_specs()
        params = ledger.latest_parameter_version() or default_params()
        engine = SchedulingEngine(nodes=nodes, pods=pods, params=params)
        return cls(nodes=nodes, pods=pods, ledger=ledger, engine=engine)

    # ------------------------------------------------------------------ #

    @property
    def params(self) -> SessionParams:
        return self.engine.params

    def spec_digest(self) -> str:
        return self.engine.spec_digest()

    def update_parameters(
        self,
        weights: Optional[Dict[str, int]] = None,
        scale_factor: Optional[int] = None,
    ) -> SessionParams:
        params = self.engine.update_params(weights=weights, scale_factor=scale_factor)
        self.ledger.record_parameter_version(params)
        # Old conclusions are stale and must not participate in the new view.
        self.ledger.reset_pod_results()
        return params

    def run_full_schedule(self, snapshot_id: Optional[str] = None) -> ScheduleReport:
        report = self.engine.run()
        self.ledger.reset_pod_results()
        for seq, decision in enumerate(report.decisions):
            self.ledger.commit_decision(seq, decision.pod_id, decision)
        if snapshot_id:
            self.ledger.save_snapshot(
                snapshot_id, self.engine.params, self.engine.spec_digest()
            )
        return report

    # ------------------------------------------------------------------ #
    # Crash-resistant incremental scheduling with UNKNOWN markers
    # ------------------------------------------------------------------ #

    def run_with_crash_marker(self, crash_before_seq: Optional[int] = None) -> List[Decision]:
        """Schedule pod-by-pod; mark the in-flight pod and every later pod UNKNOWN.

        A forced termination cannot distinguish "computing" from "not started":
        anything lacking a complete committed ledger row is unknown and must be
        rerun after restart.
        """
        self.ledger.reset_pod_results()
        capacities = self.engine.fresh_capacities()
        placements: List[Placement] = []
        decisions: List[Decision] = []
        for seq, pod in enumerate(self.pods):
            if crash_before_seq is not None and seq == crash_before_seq:
                for tail_seq, tail_pod in enumerate(self.pods[seq:], start=seq):
                    self.ledger.mark_unknown(
                        tail_seq, tail_pod.pod_id, self.params.version
                    )
                break
            decision = self.engine.schedule_one(pod, capacities, placements)
            self.ledger.commit_decision(seq, pod.pod_id, decision)
            decisions.append(decision)
            if decision.bound_node is not None:
                capacities[decision.bound_node].cpu_used += pod.cpu_request
                capacities[decision.bound_node].memory_used += pod.memory_request
                placements.append(
                    Placement(pod_id=pod.pod_id, labels=dict(pod.labels),
                              node=decision.bound_node)
                )
        return decisions

    def recover(self) -> RecoveryResult:
        return recover_and_replay(self.ledger, self.engine)

    def rerun_unknown(self) -> List[Decision]:
        return rerun_unknown_pods(self.ledger, self.engine)

    def snapshots(self) -> List[Dict[str, Any]]:
        return self.ledger.list_snapshots()

    def close(self) -> None:
        self.ledger.close()
