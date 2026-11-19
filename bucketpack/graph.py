"""Stable bucket-level dependency graph and reference validation."""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .errors import E_CYCLE, E_REF, BusinessError, fail
from .spec import Spec


def _canonical_cycle(nodes: Iterable[str]) -> tuple[str, ...]:
    items = list(nodes)
    if not items:
        return ()
    minimum = min(items)
    index = items.index(minimum)
    return tuple(items[index:] + items[:index])


@dataclass(frozen=True)
class DependencyGraph:
    parents: Mapping[str, tuple[str, ...]]
    children: Mapping[str, tuple[str, ...]]
    topo: tuple[str, ...]
    source_column_rules: Mapping[tuple[str, str], tuple[str, ...]]
    statistic_refs: Mapping[str, tuple[str, ...]]

    def descendants(self, seeds: Iterable[str]) -> tuple[str, ...]:
        """Return the seed set and all transitive dependants in stable order."""

        seen: set[str] = set()
        queue: list[str] = sorted(set(seeds))
        while queue:
            node = heapq.heappop(queue)
            if node in seen:
                continue
            seen.add(node)
            for child in self.children.get(node, ()):
                if child not in seen:
                    heapq.heappush(queue, child)
        return tuple(sorted(seen))

    def columns_triggering(self, source_id: str, column_name: str) -> tuple[str, ...]:
        return self.source_column_rules.get((source_id, column_name), ())

    def bucket_referrers(
        self,
        spec: Spec,
        bucket_ids: Iterable[str],
    ) -> list[dict[str, str]]:
        removed = set(bucket_ids)
        rule_index = {rule.id: idx for idx, rule in enumerate(spec.rules)}
        stat_index = {stat.id: idx for idx, stat in enumerate(spec.statistics)}
        refs: list[dict[str, str]] = []
        for child in sorted(removed):
            for parent_rule_id in self.children.get(child, ()):
                if parent_rule_id in removed:
                    continue
                parent_rule = spec.rule(parent_rule_id)
                assert parent_rule is not None
                idx = rule_index.get(parent_rule_id, 0)
                refs.append(
                    {
                        "referrer_type": "rule",
                        "referrer_id": parent_rule_id,
                        "field": "input.name",
                        "path": f"$.rules[{idx}].input.name",
                        "bucket": child,
                    }
                )
        for stat_id in sorted(self.statistic_refs):
            for ref in self.statistic_refs[stat_id]:
                if ref in removed:
                    idx = stat_index.get(stat_id, 0)
                    refs.append(
                        {
                            "referrer_type": "statistic",
                            "referrer_id": stat_id,
                            "field": "refs",
                            "path": f"$.statistics[{idx}].refs",
                            "bucket": ref,
                        }
                    )
        return sorted(
            refs,
            key=lambda item: (
                item["bucket"],
                item["referrer_type"],
                item["referrer_id"],
                item["path"],
            ),
        )

    def assert_deletable(self, spec: Spec, bucket_ids: Iterable[str]) -> None:
        refs = self.bucket_referrers(spec, bucket_ids)
        if refs:
            first = refs[0]
            raise BusinessError(
                code=E_REF,
                path=first["path"],
                message="bucket column is still referenced and cannot be deleted",
                details={"removed": sorted(set(bucket_ids)), "referrers": refs},
            )


def build_graph(spec: Spec) -> DependencyGraph:
    rules = {rule.id: rule for rule in spec.rules}
    parents: dict[str, set[str]] = {rule_id: set() for rule_id in rules}
    children: dict[str, set[str]] = {rule_id: set() for rule_id in rules}
    source_column_rules: dict[tuple[str, str], set[str]] = defaultdict(set)
    statistic_refs: dict[str, tuple[str, ...]] = {}
    refs_for_error: list[dict[str, str]] = []
    rule_index = {rule.id: idx for idx, rule in enumerate(spec.rules)}

    for rule_id in sorted(rules):
        rule = rules[rule_id]
        if rule.input_kind == "column":
            source = spec.source(rule.source)
            if source is None or source.column(rule.input_name) is None:
                fail(
                    E_REF,
                    f"$.rules[{rule_index[rule_id]}].input.name",
                    "input column does not exist",
                    rule_id=rule_id,
                )
            source_column_rules[(rule.source, rule.input_name)].add(rule_id)
            continue
        target = rules.get(rule.input_name)
        if target is None:
            fail(
                E_REF,
                f"$.rules[{rule_index[rule_id]}].input.name",
                "referenced bucket column does not exist",
                rule_id=rule_id,
                referenced=rule.input_name,
            )
        if target.source != rule.source:
            fail(
                E_REF,
                f"$.rules[{rule_index[rule_id]}].input.name",
                "referenced bucket column belongs to another source",
                rule_id=rule_id,
                referenced=rule.input_name,
            )
        if not target.boundaries:
            fail(
                E_REF,
                f"$.rules[{rule_index[rule.input_name]}].boundaries",
                "referenced bucket column has no boundaries",
                referenced=rule.input_name,
            )
        parents[rule_id].add(target.id)
        children[target.id].add(rule_id)

    for stat in spec.statistics:
        for ref in stat.refs:
            target = rules.get(ref)
            stat_index = next(idx for idx, item in enumerate(spec.statistics) if item.id == stat.id)
            if target is None:
                refs_for_error.append(
                    {
                        "statistic": stat.id,
                        "field": "refs",
                        "path": f"$.statistics[{stat_index}].refs",
                        "bucket": ref,
                    }
                )
                continue
            if target.source != stat.source:
                refs_for_error.append(
                    {
                        "statistic": stat.id,
                        "field": "refs",
                        "path": f"$.statistics[{stat_index}].refs",
                        "bucket": ref,
                    }
                )
        statistic_refs[stat.id] = tuple(sorted(stat.refs))
    if refs_for_error:
        first = min(refs_for_error, key=lambda item: (item["path"], item["bucket"]))
        raise BusinessError(
            code=E_REF,
            path=first["path"],
            message="statistic bucket reference does not exist or belongs to another source",
            details={"references": sorted(refs_for_error, key=lambda item: (item["path"], item["bucket"]))},
        )

    # Kahn's algorithm with a heap of ready node ids yields a canonical order.
    indegree = {rule_id: len(parents[rule_id]) for rule_id in rules}
    ready = [rule_id for rule_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    topo: list[str] = []
    while ready:
        node = heapq.heappop(ready)
        topo.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)

    if len(topo) != len(rules):
        remaining = set(rules) - set(topo)
        cycle = _find_cycle(remaining, children)
        raise BusinessError(
            code=E_CYCLE,
            path=f"$.rules[id={cycle[0]}].input.name" if cycle else "$.rules",
            message="bucket dependency graph contains a cycle",
            details={"cycle": list(cycle), "nodes": list(cycle)},
        )

    return DependencyGraph(
        parents={key: tuple(sorted(value)) for key, value in sorted(parents.items())},
        children={key: tuple(sorted(value)) for key, value in sorted(children.items())},
        topo=tuple(topo),
        source_column_rules={
            key: tuple(sorted(value)) for key, value in sorted(source_column_rules.items())
        },
        statistic_refs={key: tuple(value) for key, value in sorted(statistic_refs.items())},
    )


def _find_cycle_from(
    start: str,
    nodes: set[str],
    children: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...] | None:
    """Return one deterministic cycle that starts and ends at ``start``."""

    path: list[str] = [start]
    on_path = {start}

    def dfs(node: str) -> tuple[str, ...] | None:
        for child in sorted(children.get(node, ())):
            if child not in nodes:
                continue
            if child == start:
                return tuple(path)
            if child in on_path:
                continue
            path.append(child)
            on_path.add(child)
            found = dfs(child)
            if found is not None:
                return found
            on_path.remove(child)
            path.pop()
        return None

    return dfs(start)


def _find_cycle(nodes: set[str], children: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Find a canonical cycle within a non-empty cyclic node set."""

    for start in sorted(nodes):
        found = _find_cycle_from(start, nodes, children)
        if found is not None:
            return _canonical_cycle(found)
    return tuple(sorted(nodes))


def stable_edges(graph: DependencyGraph) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for child in sorted(graph.parents):
        for parent in graph.parents[child]:
            edges.append({"from": parent, "to": child})
    return edges
