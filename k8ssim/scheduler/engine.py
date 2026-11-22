"""Scheduling engine: parameter versioning, orchestration and staleness guard."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..constants import (
    PLUGIN_LABEL_LOCALITY,
    PLUGIN_NODE_AFFINITY,
    PLUGIN_POD_ANTI_AFFINITY,
    PLUGIN_RESOURCE_BALANCE,
)
from ..loader import validate_scale_factor, validate_weights
from ..models import Decision, Node, Pod
from ..util import digest_of
from .filter import NodeCapacity, Placement, evaluate_filter
from .plugins import score_node
from .scoring import compose_node_score, select_binding

DEFAULT_WEIGHTS = {
    PLUGIN_NODE_AFFINITY: 3000,
    PLUGIN_POD_ANTI_AFFINITY: 2000,
    PLUGIN_RESOURCE_BALANCE: 3000,
    PLUGIN_LABEL_LOCALITY: 2000,
}
DEFAULT_SCALE_FACTOR = 10_000


@dataclass(frozen=True)
class SessionParams:
    weights: Mapping[str, int]
    scale_factor: int
    version: int

    def to_dict(self) -> dict:
        return {
            "weights": dict(self.weights),
            "scale_factor": self.scale_factor,
            "version": self.version,
        }

    def content_digest(self) -> str:
        return digest_of({"weights": dict(self.weights), "scale_factor": self.scale_factor})


def default_params() -> SessionParams:
    return SessionParams(dict(DEFAULT_WEIGHTS), DEFAULT_SCALE_FACTOR, 1)


@dataclass
class ScheduleReport:
    """An immutable-ish full conclusion set for one parameter version."""

    parameter_version: int
    params_digest: str
    spec_digest: str
    decisions: List[Decision]
    bound_nodes: Dict[str, str]
    staleness: str = "current"          # current | stale
    replaced_entries: Tuple[str, ...] = ()

    def pod_order(self) -> List[str]:
        return [decision.pod_id for decision in self.decisions]

    def to_dict(self) -> dict:
        return {
            "parameter_version": self.parameter_version,
            "params_digest": self.params_digest,
            "spec_digest": self.spec_digest,
            "staleness": self.staleness,
            "replaced_entries": list(self.replaced_entries),
            "decisions": [decision.to_dict() for decision in self.decisions],
            "bound_nodes": dict(self.bound_nodes),
        }


@dataclass
class SchedulingEngine:
    nodes: Sequence[Node]
    pods: Sequence[Pod]
    params: SessionParams = field(default_factory=default_params)
    _report: Optional[ScheduleReport] = None
    _history: Dict[int, ScheduleReport] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._node_lookup = {node.name: node for node in self.nodes}
        self._capacities = {
            node.name: NodeCapacity(node.cpu, node.memory) for node in self.nodes
        }
        self._placements: List[Placement] = []

    # ------------------------------------------------------------------ #
    # Parameter changes: version increases monotonically, report replaced
    # ------------------------------------------------------------------ #

    def update_params(
        self,
        weights: Optional[Mapping[str, int]] = None,
        scale_factor: Optional[int] = None,
    ) -> SessionParams:
        new_weights = validate_weights(dict(weights or self.params.weights))
        new_scale = validate_scale_factor(
            self.params.scale_factor if scale_factor is None else scale_factor
        )
        candidate = SessionParams(new_weights, new_scale, self.params.version + 1)
        if candidate.content_digest() == self.params.content_digest():
            # No semantic change: keep the version untouched.
            return self.params
        replaced: List[str] = []
        if self._report is not None:
            self._report.staleness = "stale"
            replaced = self._report.pod_order()
            self._history[self._report.parameter_version] = self._report
        self.params = candidate
        self._report = None
        # Stash the replacement list on the next report once it is built.
        self._pending_replaced = tuple(replaced)
        return self.params

    @property
    def report(self) -> Optional[ScheduleReport]:
        return self._report

    def stale_reports(self) -> List[ScheduleReport]:
        return [report for report in self._history.values() if report.staleness == "stale"]

    # ------------------------------------------------------------------ #
    # Full (re)scheduling: conclusions are replaced as one atomic view
    # ------------------------------------------------------------------ #

    def run(self) -> ScheduleReport:
        capacities = {
            node.name: NodeCapacity(node.cpu, node.memory) for node in self.nodes
        }
        placements: List[Placement] = []
        decisions: List[Decision] = []
        bound_nodes: Dict[str, str] = {}

        for pod in self.pods:
            candidates, rejections = evaluate_filter(
                pod, self.nodes, capacities, placements, self._node_lookup
            )
            node_scores = []
            for node_name in candidates:
                node = self._node_lookup[node_name]
                outputs = score_node(
                    pod,
                    node,
                    capacities[node_name],
                    placements,
                    self._node_lookup,
                    self.params.weights,
                    self.params.scale_factor,
                )
                node_scores.append(compose_node_score(node_name, outputs))
            decision = select_binding(
                candidates,
                node_scores,
                pod.pod_id,
                self.params.version,
                rejections,
            )
            decisions.append(decision)
            if decision.bound_node is not None:
                bound = decision.bound_node
                capacities[bound].cpu_used += pod.cpu_request
                capacities[bound].memory_used += pod.memory_request
                placements.append(
                    Placement(pod_id=pod.pod_id, labels=dict(pod.labels), node=bound)
                )
                bound_nodes[pod.pod_id] = bound

        replaced = getattr(self, "_pending_replaced", ())
        if hasattr(self, "_pending_replaced"):
            del self._pending_replaced
        report = ScheduleReport(
            parameter_version=self.params.version,
            params_digest=self.params.content_digest(),
            spec_digest=self.spec_digest(),
            decisions=decisions,
            bound_nodes=bound_nodes,
            staleness="current",
            replaced_entries=replaced,
        )
        self._capacities = capacities
        self._placements = placements
        self._report = report
        return report

    def spec_digest(self) -> str:
        return digest_of(
            {
                "nodes": [node.to_dict() for node in sorted(self.nodes, key=lambda n: n.name)],
                "pods": [pod.to_dict() for pod in self.pods],
            }
        )

    # ------------------------------------------------------------------ #
    # Single-pod scheduling used by crash/replay flows
    # ------------------------------------------------------------------ #

    def schedule_one(
        self,
        pod: Pod,
        capacities: Mapping[str, NodeCapacity],
        placements: Sequence[Placement],
    ) -> Decision:
        candidates, rejections = evaluate_filter(
            pod, self.nodes, capacities, placements, self._node_lookup
        )
        node_scores = []
        for node_name in candidates:
            node = self._node_lookup[node_name]
            outputs = score_node(
                pod,
                node,
                capacities[node_name],
                placements,
                self._node_lookup,
                self.params.weights,
                self.params.scale_factor,
            )
            node_scores.append(compose_node_score(node_name, outputs))
        return select_binding(
            candidates, node_scores, pod.pod_id, self.params.version, rejections
        )

    def fresh_capacities(self) -> Dict[str, NodeCapacity]:
        return {
            node.name: NodeCapacity(node.cpu, node.memory) for node in self.nodes
        }
