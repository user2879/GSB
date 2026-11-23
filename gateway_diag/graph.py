from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable

from .errors import E_CYCLE, GatewayError
from .models import Spec
from .routing import Router
from .util import digest_value


def dependencies(spec: Spec) -> dict[str, tuple[str, ...]]:
    return {rule_id: spec.rules[rule_id].dependency_ids for rule_id in spec.sorted_rule_ids}


def dependents(spec: Spec) -> dict[str, tuple[str, ...]]:
    reverse: dict[str, list[str]] = {rule_id: [] for rule_id in spec.sorted_rule_ids}
    for rule_id, deps in dependencies(spec).items():
        for dep in deps:
            reverse[dep].append(rule_id)
    return {key: tuple(sorted(values)) for key, values in reverse.items()}


def dependency_graph_payload(spec: Spec) -> dict[str, Any]:
    nodes = []
    for rule_id in spec.sorted_rule_ids:
        rule = spec.rules[rule_id]
        nodes.append(
            {
                "id": rule_id,
                "batch_size": rule.batch_size,
                "stable_backend": rule.stable_backend,
                "canary_target": rule.canary_target.payload(),
                "depends_on": list(rule.dependency_ids),
            }
        )
    return {"schema": "dependency-graph.v1", "nodes": nodes}


def dependency_graph_digest(spec: Spec) -> str:
    return digest_value(dependency_graph_payload(spec))


def detect_cycle(spec: Spec) -> list[str] | None:
    """Return a deterministic cycle path including the repeated start node."""
    deps = dependencies(spec)
    state: dict[str, int] = {rule_id: 0 for rule_id in spec.sorted_rule_ids}
    stack: list[str] = []
    position: dict[str, int] = {}

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        position[node] = len(stack)
        stack.append(node)
        for dep in deps[node]:
            if state[dep] == 0:
                found = visit(dep)
                if found:
                    return found
            elif state[dep] == 1:
                start = position[dep]
                return stack[start:] + [dep]
        stack.pop()
        position.pop(node, None)
        state[node] = 2
        return None

    for rule_id in spec.sorted_rule_ids:
        if state[rule_id] == 0:
            found = visit(rule_id)
            if found:
                return found
    return None


def ensure_acyclic(spec: Spec) -> None:
    cycle = detect_cycle(spec)
    if cycle:
        raise GatewayError(
            E_CYCLE,
            "rule dependency graph contains a cycle",
            rule_id=cycle[0],
            field="depends_on",
            details={"cycle": cycle},
        )


def topological_order(spec: Spec, rule_ids: Iterable[str] | None = None) -> list[str]:
    selected = set(spec.sorted_rule_ids if rule_ids is None else rule_ids)
    deps = dependencies(spec)
    indegree = {rule_id: 0 for rule_id in selected}
    reverse = {rule_id: [] for rule_id in selected}
    for rule_id in sorted(selected):
        for dep in deps[rule_id]:
            if dep in selected:
                indegree[rule_id] += 1
                reverse[dep].append(rule_id)
    ready = deque(sorted(rule_id for rule_id, degree in indegree.items() if degree == 0))
    result: list[str] = []
    while ready:
        node = ready.popleft()
        result.append(node)
        for child in sorted(reverse[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                # Insert while preserving deterministic lexicographic tie breaking.
                ready.append(child)
        # ``deque`` append order can depend on a parent's child order; sort
        # before the next pop to make ties independent of declaration order.
        ready = deque(sorted(ready))
    if len(result) != len(selected):
        cycle = detect_cycle(spec)
        raise GatewayError(
            E_CYCLE,
            "subset dependency graph contains a cycle",
            rule_id=cycle[0] if cycle else None,
            field="depends_on",
            details={"cycle": cycle or []},
        )
    return result


def initial_invalidation(spec: Spec) -> dict[str, Any]:
    order = topological_order(spec)
    steps = []
    for index, rule_id in enumerate(order, start=1):
        steps.append(
            {
                "step": index,
                "rule_id": rule_id,
                "code": "E_STALE",
                "basis": {"type": "initial_build", "field": None},
            }
        )
    return {
        "changed_backends": [],
        "added_backends": [],
        "removed_backends": [],
        "affected_backends": list(spec.sorted_backend_ids),
        "directly_invalidated": list(order),
        "invalidated_rules": steps,
        "removed_rules": [],
        "added_rules": list(order),
        "unaffected_rules": [],
        "replacement_rules": list(order),
        "graph_digest_before": None,
        "graph_digest_after": dependency_graph_digest(spec),
    }


def _request_assignments(spec: Spec) -> dict[str, set[str]]:
    assignments: dict[str, set[str]] = {}
    if not spec.requests:
        return assignments
    router = Router(spec)
    for request in spec.requests:
        rule = router.matching_rule(request)
        assignments.setdefault(rule.id, set()).add(request.request_id)
    return assignments


def _rule_field_changes(before_rule, after_rule) -> list[dict[str, Any]]:
    old = before_rule.comparable_payload()
    new = after_rule.comparable_payload()
    changes: list[dict[str, Any]] = []
    for field in sorted(set(old) | set(new)):
        old_value = old.get(field)
        new_value = new.get(field)
        if old_value != new_value:
            changes.append(
                {
                    "type": "rule_field_changed",
                    "field": field,
                    "old": old_value,
                    "new": new_value,
                }
            )
    return changes


def compute_invalidation(before: Spec, after: Spec) -> dict[str, Any]:
    """Compute the deterministic closure and a replayable propagation trail."""
    before_rules = set(before.rules)
    after_rules = set(after.rules)
    added = sorted(after_rules - before_rules)
    removed = sorted(before_rules - after_rules)

    direct: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule_id in sorted(before_rules & after_rules):
        for change in _rule_field_changes(before.rules[rule_id], after.rules[rule_id]):
            direct[rule_id].append(change)
    for rule_id in added:
        direct[rule_id].append({"type": "rule_added", "field": "id", "new": rule_id})

    before_assignments = _request_assignments(before)
    after_assignments = _request_assignments(after)
    for rule_id in sorted(after_rules):
        old_ids = before_assignments.get(rule_id, set())
        new_ids = after_assignments.get(rule_id, set())
        if old_ids != new_ids:
            direct[rule_id].append(
                {
                    "type": "request_assignment_changed",
                    "field": "request_ids",
                    "added": sorted(new_ids - old_ids),
                    "removed": sorted(old_ids - new_ids),
                }
            )

    backend_ids = sorted(set(before.backends) | set(after.backends))
    changed_backends: list[str] = []
    added_backends: list[str] = []
    removed_backends: list[str] = []
    backend_changes: dict[str, dict[str, Any]] = {}
    for backend_id in backend_ids:
        old = before.backends.get(backend_id)
        new = after.backends.get(backend_id)
        if old is None:
            added_backends.append(backend_id)
            changed_backends.append(backend_id)
            backend_changes[backend_id] = {"type": "backend_added"}
        elif new is None:
            removed_backends.append(backend_id)
            changed_backends.append(backend_id)
            backend_changes[backend_id] = {"type": "backend_removed"}
        elif old.declaration_payload != new.declaration_payload:
            changed_backends.append(backend_id)
            backend_changes[backend_id] = {
                "type": "backend_parameters_changed",
                "old_param_digest": old.param_digest,
                "new_param_digest": new.param_digest,
                "old": old.declaration_payload,
                "new": new.declaration_payload,
            }
    for rule_id in sorted(after_rules):
        touched = sorted(set(after.rules[rule_id].direct_backend_ids) & set(changed_backends))
        for backend_id in touched:
            direct[rule_id].append(
                {
                    "type": "backend_changed",
                    "field": "backend",
                    "backend_id": backend_id,
                    **backend_changes[backend_id],
                }
            )

    # Directly changed rules are seeds.  Every downstream dependent is stale,
    # regardless of whether its own declaration changed.
    reverse = dependents(after)
    invalid: set[str] = set(direct)
    queue = deque(sorted(invalid))
    propagation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    while queue:
        upstream = queue.popleft()
        for child in reverse.get(upstream, ()):  # deterministic tuple order
            propagation[child].append(
                {"type": "dependency_invalidated", "upstream_rule": upstream}
            )
            if child not in invalid:
                invalid.add(child)
                queue.append(child)

    order = topological_order(after, invalid)
    steps: list[dict[str, Any]] = []
    for index, rule_id in enumerate(order, start=1):
        reasons = sorted(
            direct.get(rule_id, []),
            key=lambda reason: (
                str(reason.get("type")),
                str(reason.get("field")),
                str(reason.get("backend_id", "")),
            ),
        )
        upstreams = sorted(
            {reason["upstream_rule"] for reason in propagation.get(rule_id, [])}
        )
        if reasons:
            basis = reasons[0]
        else:
            basis = {
                "type": "dependency_invalidated",
                "upstream_rule": upstreams[0] if upstreams else None,
            }
        steps.append(
            {
                "step": index,
                "rule_id": rule_id,
                "code": "E_STALE",
                "basis": basis,
                "all_bases": reasons
                + [
                    {"type": "dependency_invalidated", "upstream_rule": upstream}
                    for upstream in upstreams
                ],
            }
        )

    affected_backends = set(changed_backends)
    affected_rules = set(invalid) | set(removed)
    for rule_id in sorted(affected_rules):
        rule = after.rules.get(rule_id) or before.rules.get(rule_id)
        if rule is not None:
            affected_backends.update(rule.direct_backend_ids)

    unaffected = sorted(after_rules - invalid)
    replacement = sorted(set(invalid) | set(removed))
    return {
        "changed_backends": sorted(changed_backends),
        "added_backends": sorted(added_backends),
        "removed_backends": sorted(removed_backends),
        "affected_backends": sorted(affected_backends),
        "directly_invalidated": sorted(direct),
        "invalidated_rules": steps,
        "removed_rules": removed,
        "added_rules": added,
        "unaffected_rules": unaffected,
        "replacement_rules": replacement,
        "graph_digest_before": dependency_graph_digest(before),
        "graph_digest_after": dependency_graph_digest(after),
    }
