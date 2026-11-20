"""Dependency graph for facts, derived node statistics and pod conclusions."""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable

from .errors import BusinessError, E_GRAPH
from .model import Node, Pod, PluginConfig
from .util import sha256_digest


def fact_plugin_weights() -> str:
    return "fact:plugin:weights"


def fact_node_label(node: str, key: str) -> str:
    return f"fact:node:{node}:label:{key}"


def fact_node_taints(node: str) -> str:
    return f"fact:node:{node}:taints"


def fact_node_capacity(node: str) -> str:
    return f"fact:node:{node}:capacity"


def fact_node_images(node: str) -> str:
    return f"fact:node:{node}:images"


def fact_pod_labels(pod: str) -> str:
    return f"fact:pod:{pod}:labels"


def fact_pod_affinity(pod: str) -> str:
    return f"fact:pod:{pod}:affinity"


def fact_pod_resources(pod: str) -> str:
    return f"fact:pod:{pod}:resources"


def fact_pod_tolerations(pod: str) -> str:
    return f"fact:pod:{pod}:tolerations"


def fact_pod_image(pod: str) -> str:
    return f"fact:pod:{pod}:image"


def stat_label_match(pod: str, node: str) -> str:
    return f"stat:label-match:{pod}:{node}"


def stat_taint_gate(pod: str, node: str) -> str:
    return f"stat:taint-gate:{pod}:{node}"


def stat_resource_remaining(pod: str, node: str) -> str:
    return f"stat:resource-remaining:{pod}:{node}"


def stat_anti_count(pod: str, node: str) -> str:
    return f"stat:anti-count:{pod}:{node}"


def stat_image_locality(pod: str, node: str) -> str:
    return f"stat:image-locality:{pod}:{node}"


def result_pod(pod: str) -> str:
    return f"result:pod:{pod}"


STAT_KINDS: dict[str, str] = {
    "stat:label-match:": "label_match_set",
    "stat:taint-gate:": "taint_gate",
    "stat:resource-remaining:": "resource_remaining",
    "stat:anti-count:": "same_label_pod_count",
    "stat:image-locality:": "image_locality",
}


def kind_for(node_id: str) -> str:
    if node_id.startswith("fact:"):
        return "fact"
    if node_id.startswith("result:pod:"):
        return "pod_conclusion"
    for prefix, kind in STAT_KINDS.items():
        if node_id.startswith(prefix):
            return kind
    return "unknown"


@dataclass
class DependencyGraph:
    nodes: dict[str, str] = field(default_factory=dict)
    order: dict[str, int] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=dict)
    reverse: dict[str, set[str]] = field(default_factory=dict)

    def add_node(self, node_id: str, kind: str) -> None:
        if node_id in self.nodes:
            return
        self.nodes[node_id] = kind
        self.order[node_id] = len(self.order)
        self.edges[node_id] = set()
        self.reverse[node_id] = set()

    def add_edge(self, dependency: str, dependent: str) -> None:
        if dependency not in self.nodes or dependent not in self.nodes:
            raise BusinessError(
                E_GRAPH,
                "dependency edge references a missing graph node",
                details={"dependency": dependency, "dependent": dependent},
            )
        self.edges[dependency].add(dependent)
        self.reverse[dependent].add(dependency)

    def validate(self) -> dict[str, int]:
        missing: list[dict[str, str]] = []
        for dependency, dependents in self.edges.items():
            for dependent in sorted(dependents):
                if dependent not in self.nodes:
                    missing.append({"dependency": dependency, "dependent": dependent})
        if missing:
            raise BusinessError(E_GRAPH, "dependency graph has unresolved references", details={"missing": missing})
        ordered = self.topological_order(self.nodes)
        if len(ordered) != len(self.nodes):
            cyclic = sorted(node for node in self.nodes if node not in set(ordered))
            raise BusinessError(E_GRAPH, "dependency graph contains a cycle", details={"unresolved": cyclic})
        return {
            "nodes": len(self.nodes),
            "edges": sum(len(dependents) for dependents in self.edges.values()),
            "facts": sum(1 for kind in self.nodes.values() if kind == "fact"),
            "stats": sum(1 for kind in self.nodes.values() if kind.startswith(("label_", "taint_", "resource_", "same_", "image_"))),
            "pod_conclusions": sum(1 for kind in self.nodes.values() if kind == "pod_conclusion"),
        }

    def _heap_key(self, node_id: str) -> tuple[int, str]:
        return self.order.get(node_id, 1 << 60), node_id

    def topological_order(self, subset: Iterable[str] | None = None) -> list[str]:
        selected = set(self.nodes if subset is None else subset)
        indegree = {node_id: 0 for node_id in selected}
        for dependency in selected:
            for dependent in self.edges.get(dependency, ()):
                if dependent in selected:
                    indegree[dependent] += 1
        heap: list[tuple[int, str]] = []
        for node_id, degree in indegree.items():
            if degree == 0:
                heapq.heappush(heap, self._heap_key(node_id))
        result: list[str] = []
        while heap:
            _, node_id = heapq.heappop(heap)
            result.append(node_id)
            for dependent in sorted(self.edges.get(node_id, ())):
                if dependent not in selected:
                    continue
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    heapq.heappush(heap, self._heap_key(dependent))
        return result

    def descendants(self, roots: Iterable[str]) -> set[str]:
        root_list = sorted(set(roots))
        missing = [root for root in root_list if root not in self.nodes]
        if missing:
            raise BusinessError(E_GRAPH, "invalidation root is absent from the dependency graph", details={"roots": missing})
        seen = set(root_list)
        stack = list(reversed(root_list))
        while stack:
            node_id = stack.pop()
            for dependent in sorted(self.edges.get(node_id, ()), reverse=True):
                if dependent not in seen:
                    seen.add(dependent)
                    stack.append(dependent)
        return seen

    def invalidation_report(self, roots: Iterable[str], source: str) -> dict[str, object]:
        root_list = sorted(set(roots))
        selected = self.descendants(root_list)
        ordered = self.topological_order(selected)
        root_set = set(root_list)
        steps: list[dict[str, object]] = []
        stats: list[str] = []
        pods: list[str] = []
        for index, node_id in enumerate(ordered, start=1):
            parents = sorted(parent for parent in self.reverse.get(node_id, ()) if parent in selected)
            if node_id in root_set:
                triggers = [f"external:{source}"]
            else:
                triggers = parents
            kind = self.nodes.get(node_id, kind_for(node_id))
            steps.append(
                {
                    "step": index,
                    "id": node_id,
                    "kind": kind,
                    "triggered_by": triggers,
                }
            )
            if kind != "fact" and kind != "pod_conclusion":
                stats.append(node_id)
            if kind == "pod_conclusion":
                pods.append(node_id.removeprefix("result:pod:"))
        return {
            "source": source,
            "roots": root_list,
            "steps": steps,
            "stats": stats,
            "pods": pods,
            "all_ids": ordered,
        }

    def signature(self) -> str:
        payload = {
            "nodes": [[node_id, self.nodes[node_id]] for node_id in sorted(self.nodes)],
            "edges": [[dependency, sorted(dependents)] for dependency, dependents in sorted(self.edges.items())],
        }
        return sha256_digest(payload)


def _selector_dependencies(pod: Pod) -> list[str]:
    deps = [fact_pod_affinity(pod.pod_id)]
    return deps


def build_graph(nodes: list[Node], pods: list[Pod], plugins: PluginConfig) -> DependencyGraph:
    """Build the complete acyclic dependency graph for the current manifest.

    Edges point from a fact/stat/conclusion to data derived from it. The pod
    queue is respected: an earlier pod conclusion feeds later resource and
    anti-affinity statistics, never the reverse.
    """
    graph = DependencyGraph()
    nodes_by_name = {node.name: node for node in nodes}
    node_names = sorted(nodes_by_name)

    graph.add_node(fact_plugin_weights(), "fact")

    selector_keys = {key for pod in pods for key in pod.node_selector_keys()}
    label_universe = selector_keys | {key for node in nodes for key in node.labels}
    for node_name in node_names:
        node = nodes_by_name[node_name]
        for key in sorted(label_universe):
            graph.add_node(fact_node_label(node_name, key), "fact")
        graph.add_node(fact_node_taints(node_name), "fact")
        graph.add_node(fact_node_capacity(node_name), "fact")
        graph.add_node(fact_node_images(node_name), "fact")

    for pod in pods:
        graph.add_node(fact_pod_labels(pod.pod_id), "fact")
        graph.add_node(fact_pod_affinity(pod.pod_id), "fact")
        graph.add_node(fact_pod_resources(pod.pod_id), "fact")
        graph.add_node(fact_pod_tolerations(pod.pod_id), "fact")
        graph.add_node(fact_pod_image(pod.pod_id), "fact")

    for pod_index, pod in enumerate(pods):
        earlier = pods[:pod_index]
        node_keys: set[str] = set()
        if pod.node_affinity_hard is not None:
            node_keys |= pod.node_affinity_hard.referenced_keys()
        for rule in pod.node_affinity_soft:
            node_keys |= rule.selector.referenced_keys()
        anti_topology_keys = {rule.topology_key for rule in pod.anti_affinity_hard + pod.anti_affinity_soft}

        for node_name in node_names:
            label_match = stat_label_match(pod.pod_id, node_name)
            taint_gate = stat_taint_gate(pod.pod_id, node_name)
            remaining = stat_resource_remaining(pod.pod_id, node_name)
            anti_count = stat_anti_count(pod.pod_id, node_name)
            image_locality = stat_image_locality(pod.pod_id, node_name)
            for node_id, kind in (
                (label_match, "label_match_set"),
                (taint_gate, "taint_gate"),
                (remaining, "resource_remaining"),
                (anti_count, "same_label_pod_count"),
                (image_locality, "image_locality"),
            ):
                graph.add_node(node_id, kind)

            for key in sorted(node_keys):
                graph.add_edge(fact_node_label(node_name, key), label_match)
            graph.add_edge(fact_pod_affinity(pod.pod_id), label_match)

            graph.add_edge(fact_node_taints(node_name), taint_gate)
            graph.add_edge(fact_pod_tolerations(pod.pod_id), taint_gate)

            graph.add_edge(fact_node_capacity(node_name), remaining)
            graph.add_edge(fact_pod_resources(pod.pod_id), remaining)
            for prior in earlier:
                graph.add_edge(result_pod(prior.pod_id), remaining)

            for topology_key in sorted(anti_topology_keys):
                # A prior pod may have been placed on any node.  Its topology
                # label therefore participates in this node's anti-count.
                for topology_node in node_names:
                    graph.add_edge(fact_node_label(topology_node, topology_key), anti_count)
            graph.add_edge(fact_pod_affinity(pod.pod_id), anti_count)
            for prior in earlier:
                graph.add_edge(fact_pod_labels(prior.pod_id), anti_count)
                graph.add_edge(result_pod(prior.pod_id), anti_count)

            graph.add_edge(fact_node_images(node_name), image_locality)
            graph.add_edge(fact_pod_image(pod.pod_id), image_locality)

        conclusion = result_pod(pod.pod_id)
        graph.add_node(conclusion, "pod_conclusion")
        for node_name in node_names:
            for stat_id in (
                stat_label_match(pod.pod_id, node_name),
                stat_taint_gate(pod.pod_id, node_name),
                stat_resource_remaining(pod.pod_id, node_name),
                stat_anti_count(pod.pod_id, node_name),
                stat_image_locality(pod.pod_id, node_name),
            ):
                graph.add_edge(stat_id, conclusion)
        graph.add_edge(fact_plugin_weights(), conclusion)

    graph.validate()
    return graph
