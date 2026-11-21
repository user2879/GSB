"""Directed dependency graph and invalidation propagation.

Tenant arbitration outcomes, the selectable instance set and health
statistics are all derived from probe verdicts and instance weights.
The graph below makes those derivations explicit; every edge is
validated for referential integrity. When an instance health
transition, an instance-set change or a rule-version evolution occurs,
the affected tenants and requests are derived by walking dependents in
stable (sorted) order, emitting the per-step propagation basis.
"""

from __future__ import annotations

from collections import deque

from .errors import SpecError
from .spec import Request, Spec


class DependencyGraph:
    def __init__(self) -> None:
        self.deps = {}

    def add(self, node, depends_on=()):
        self.deps.setdefault(node, set()).update(depends_on)
        for dep in depends_on:
            self.deps.setdefault(dep, set())

    def validate(self):
        for node in sorted(self.deps):
            for dep in sorted(self.deps[node]):
                if dep not in self.deps:
                    raise SpecError(
                        f"dependency graph integrity: node {node!r} references missing node {dep!r}",
                        field="dependency_graph",
                    )

    def dependents_index(self):
        index = {node: set() for node in self.deps}
        for node, deps in self.deps.items():
            for dep in deps:
                index[dep].add(node)
        return index


def build_graph(spec, requests):
    graph = DependencyGraph()
    graph.add("spec")
    graph.add("rule", ["spec"])
    graph.add("probe_policy", ["spec"])
    graph.add("stats")
    for iid in sorted(spec.instances):
        graph.add(f"instance:{iid}", ["spec"])
        graph.add(f"health:{iid}", [f"instance:{iid}", "probe_policy"])
        graph.deps["stats"].add(f"health:{iid}")
        graph.deps["stats"].add(f"instance:{iid}")
    for tid in sorted(spec.tenants):
        iid = spec.tenant_instance[tid]
        graph.add(f"tenant:{tid}", [f"instance:{iid}", f"health:{iid}", "rule"])
    for req in requests:
        graph.add(f"request:{req.request_id}", [f"tenant:{req.tenant_id}"])
    graph.validate()
    return graph


def affected_set(graph, changed_nodes):
    """BFS over dependents from the changed nodes. Returns
    (affected_sorted, steps) where steps are (node, via) pairs in
    discovery order."""
    index = graph.dependents_index()
    seen = set(changed_nodes)
    steps = []
    queue = deque(sorted(changed_nodes))
    while queue:
        node = queue.popleft()
        for dependent in sorted(index.get(node, ())):
            if dependent not in seen:
                seen.add(dependent)
                steps.append((dependent, node))
                queue.append(dependent)
    affected = sorted(seen - set(changed_nodes))
    return affected, steps


def affected_for_transition(spec, requests, instance_id):
    """Affected tenants / requests / stats for one health transition."""
    graph = build_graph(spec, requests)
    return affected_set(graph, [f"health:{instance_id}"])


def verify_unaffected_unchanged(outcomes_before, outcomes_after, affected):
    """Check that every request NOT in the affected set has a byte-identical
    outcome + basis sequence across the two runs. Returns the sorted list
    of violating request ids (empty when the invariant holds)."""
    affected_requests = {n.split(":", 1)[1] for n in affected if n.startswith("request:")}
    violations = []
    for request_id, before in outcomes_before.items():
        if request_id in affected_requests:
            continue
        after = outcomes_after.get(request_id)
        if after is None:
            violations.append(request_id)
            continue
        if (before.outcome, before.basis_blob()) != (after.outcome, after.basis_blob()):
            violations.append(request_id)
    return sorted(violations)
