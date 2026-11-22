"""Filter stage: hard constraints evaluated through a fixed gate sequence.

Gate order (contract): taint -> node affinity hard -> pod anti-affinity hard ->
resource headroom. Nodes are always traversed sorted by name, and each node
records only the first gate that rejects it, so identical inputs always yield
identical candidate sets and rejection diagnostics regardless of file order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple, List

from ..models import Node, Pod, Rejection
from .match import all_match, matches_requirement, taints_tolerated, toleration_covers

GATE_TAINT = "taint"
GATE_NODE_AFFINITY = "node_affinity_hard"
GATE_POD_ANTI_AFFINITY = "pod_anti_affinity_hard"
GATE_RESOURCE = "resource"

GATE_ORDER = (
    GATE_TAINT,
    GATE_NODE_AFFINITY,
    GATE_POD_ANTI_AFFINITY,
    GATE_RESOURCE,
)


@dataclass
class NodeCapacity:
    cpu_total: int
    memory_total: int
    cpu_used: int = 0
    memory_used: int = 0

    @property
    def cpu_available(self) -> int:
        return self.cpu_total - self.cpu_used

    @property
    def memory_available(self) -> int:
        return self.memory_total - self.memory_used


@dataclass(frozen=True)
class Placement:
    pod_id: str
    labels: Mapping[str, str]
    node: str


def _anti_affinity_conflicts(
    pod: Pod,
    node: Node,
    placements: Sequence[Placement],
    node_lookup: Mapping[str, Node],
) -> str:
    anti = pod.pod_anti_affinity
    if not anti.required:
        return ""
    topology_key = anti.required_topology_key
    own_domain = node.labels.get(topology_key)
    for placed in placements:
        placed_node = node_lookup[placed.node]
        if topology_key == "kubernetes.io/hostname":
            same_domain = placed.node == node.name
        else:
            same_domain = placed_node.labels.get(topology_key) == own_domain
        if not same_domain:
            continue
        if all_match(placed.labels, anti.required):
            return (
                f"co-located pod {placed.pod_id} matches required anti-affinity "
                f"on topology key {topology_key!r}"
            )
    return ""


def evaluate_filter(
    pod: Pod,
    nodes: Sequence[Node],
    capacities: Mapping[str, NodeCapacity],
    placements: Sequence[Placement],
    node_lookup: Mapping[str, Node],
) -> Tuple[List[str], List[Rejection]]:
    """Return feasible node names (sorted) and per-node rejection records."""
    candidates: List[str] = []
    rejections: List[Rejection] = []
    for node in sorted(nodes, key=lambda item: item.name):
        rejection = _evaluate_node(pod, node, capacities[node.name], placements, node_lookup)
        if rejection is None:
            candidates.append(node.name)
        else:
            rejections.append(rejection)
    return candidates, rejections


def _evaluate_node(
    pod: Pod,
    node: Node,
    capacity: NodeCapacity,
    placements: Sequence[Placement],
    node_lookup: Mapping[str, Node],
):
    # Gate 1: taints / tolerations.
    if not taints_tolerated(node.taints, pod.tolerations):
        uncovered = [
            taint.key
            for taint in node.taints
            if not any(toleration_covers(tol, taint) for tol in pod.tolerations)
        ]
        return Rejection(
            node=node.name,
            gate=GATE_TAINT,
            reason="untolerated taint",
            detail=f"taint keys without toleration: {','.join(uncovered) or '?'}",
        )

    # Gate 2: node affinity hard constraints.
    if not all_match(node.labels, pod.node_affinity.required):
        failed = next(
            req.key
            for req in pod.node_affinity.required
            if not matches_requirement(node.labels, req)
        )
        return Rejection(
            node=node.name,
            gate=GATE_NODE_AFFINITY,
            reason="required node affinity not satisfied",
            detail=f"first failed requirement key: {failed}",
        )

    # Gate 3: pod anti-affinity hard constraints.
    conflict = _anti_affinity_conflicts(pod, node, placements, node_lookup)
    if conflict:
        return Rejection(
            node=node.name,
            gate=GATE_POD_ANTI_AFFINITY,
            reason="required pod anti-affinity conflict",
            detail=conflict,
        )

    # Gate 4: resource headroom (CPU then memory, both must fit).
    if capacity.cpu_available < pod.cpu_request:
        return Rejection(
            node=node.name,
            gate=GATE_RESOURCE,
            reason="insufficient cpu",
            detail=f"need {pod.cpu_request}, available {capacity.cpu_available}",
        )
    if capacity.memory_available < pod.memory_request:
        return Rejection(
            node=node.name,
            gate=GATE_RESOURCE,
            reason="insufficient memory",
            detail=f"need {pod.memory_request}, available {capacity.memory_available}",
        )
    return None
