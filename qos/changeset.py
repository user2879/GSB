"""增量变更集：原子应用与失效传播推导。

变更集支持：新增规则、修改规则、删除规则、删除依赖边。
应用是原子的：任何前置校验失败（E_REF、E_RULE、E_DEPEND）都不会改变现状。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import RefError, RuleError
from .graph import DependencyGraph, TraceEntry
from .models import Rule, parse_rules, validate_rule_dicts


@dataclass
class ChangeSet:
    upserts: list[dict[str, Any]] = field(default_factory=list)
    delete_rule_ids: list[str] = field(default_factory=list)
    remove_edges: list[dict[str, str]] = field(default_factory=list)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ChangeSet":
        return ChangeSet(
            upserts=list(data.get("upserts", [])),
            delete_rule_ids=list(data.get("delete_rule_ids", [])),
            remove_edges=list(data.get("remove_edges", [])),
        )


@dataclass
class ApplyResult:
    rules: list[Rule]
    graph: DependencyGraph
    invalidation_sequence: list[str]
    trace: list[TraceEntry]
    change_summary: dict[str, list[str]]  # added/modified/deleted/edge_removed


def _check_removable(graph: DependencyGraph, rule_ids: list[str]) -> None:
    blockers: dict[str, list[str]] = {}
    for rid in rule_ids:
        if rid not in graph.rules:
            raise RuleError(f"待删除规则 {rid} 不存在",
                            details={"rule_id": rid, "field_path": "delete_rule_ids"})
        referrers = sorted(graph.downstream.get(rid, set()))
        if referrers:
            blockers[rid] = referrers
    if blockers:
        raise RefError(
            "存在仍被其他规则引用的规则，拒绝删除",
            details={"blocked": blockers},
        )


def apply_changeset(rules: list[Rule], changeset: ChangeSet) -> ApplyResult:
    """在现有规则集上原子应用变更集，并推导失效集合。"""
    old_graph = DependencyGraph(rules)
    old_graph.validate_all()

    # 1) 删除前置校验（E_REF）。
    _check_removable(old_graph, changeset.delete_rule_ids)

    # 2) upsert 内容字段级校验（与全量加载同一套强校验）；空 upsert 合法。
    upsert_violations = (validate_rule_dicts(changeset.upserts)
                         if changeset.upserts else [])
    if upsert_violations:
        raise RuleError(
            "变更集规则字段校验失败，整体拒绝应用",
            details={"violations": [v.to_dict() for v in upsert_violations]},
        )

    # 3) 在副本上构造目标规则集。
    working: dict[str, Rule] = {r.rule_id: r for r in rules}
    added: list[str] = []
    modified: list[str] = []
    for new_rule in parse_rules(changeset.upserts):
        (added if new_rule.rule_id not in working else modified).append(new_rule.rule_id)
        working[new_rule.rule_id] = new_rule

    edge_removed: list[str] = []
    for edge in changeset.remove_edges:
        child = edge.get("from_rule")
        parent = edge.get("upstream")
        if child not in working or not isinstance(parent, str):
            raise RuleError("依赖边删除目标不存在",
                            details={"edge": edge, "field_path": "remove_edges"})
        current = list(working[child].upstream)
        if parent not in current:
            raise RuleError(f"规则 {child} 不存在到上游 {parent} 的依赖边",
                            details={"edge": edge, "field_path": "remove_edges"})
        current.remove(parent)
        old = working[child]
        working[child] = Rule(
            rule_id=old.rule_id, dst_addr=old.dst_addr,
            port_prefixes=old.port_prefixes, rate_kbps=old.rate_kbps,
            burst_bytes=old.burst_bytes,
            queue_limit_packets=old.queue_limit_packets,
            upstream=tuple(current), actions=old.actions)
        edge_removed.append(f"{parent}->{child}")
        if child not in modified and child not in added:
            modified.append(child)

    for rid in changeset.delete_rule_ids:
        working.pop(rid)

    candidate = list(working.values())
    new_graph = DependencyGraph(candidate)
    ref_violations = new_graph.validate_references()
    if ref_violations:
        raise RuleError(
            "变更后上游引用失效，整体拒绝应用",
            details={"violations": [v.to_dict() for v in ref_violations]},
        )
    new_graph.ensure_acyclic()  # E_DEPEND

    # 4) 失效传播。
    deleted = set(changeset.delete_rule_ids)
    changed_direct = set(added) | set(modified)
    trigger = {**{rid: f"add:{rid}" for rid in added},
               **{rid: f"modify:{rid}" for rid in modified}}

    # 旧图上从“所有变更根（含待删除节点）”做 BFS，得到稳定的直接父来源，
    # 这样删除引发的间接下游也能准确标注触发源。
    roots = changed_direct | deleted
    old_parent: dict[str, str] = {}
    frontier = sorted(roots)
    seen = set(roots)
    while frontier:
        node = frontier.pop(0)
        for child in sorted(old_graph.downstream.get(node, ())):  # type: ignore[arg-type]
            if child not in seen:
                seen.add(child)
                old_parent[child] = node
                frontier.append(child)

    propagated_old = seen - roots
    # 新图上对存活的直接变更点再求一次传递闭包（覆盖改边后的新下游关系）。
    propagated_new = new_graph.transitive_downstream(changed_direct)
    invalid_set = (changed_direct | propagated_new | propagated_old) - deleted
    sequence = [rid for rid in new_graph.topo_order() if rid in invalid_set]
    trace = _build_trace(sequence, changed_direct, trigger, old_parent,
                         new_graph, deleted)

    summary = {
        "added": sorted(added),
        "modified": sorted(modified),
        "deleted": sorted(deleted),
        "edge_removed": sorted(edge_removed),
    }
    return ApplyResult(candidate, new_graph, sequence, trace, summary)


def _build_trace(sequence: list[str], changed_direct: set[str],
                 trigger: dict[str, str], old_parent: dict[str, str],
                 new_graph: DependencyGraph, deleted: set[str]) -> list[TraceEntry]:
    # 新图 BFS 父来源（用于修改/新增引发的传播链）。
    new_parent: dict[str, str] = {}
    frontier = sorted(changed_direct)
    seen = set(changed_direct)
    while frontier:
        node = frontier.pop(0)
        for child in sorted(new_graph.downstream.get(node, ())):  # type: ignore[arg-type]
            if child not in seen:
                seen.add(child)
                new_parent[child] = node
                frontier.append(child)

    trace: list[TraceEntry] = []
    for step, rid in enumerate(sequence, start=1):
        if rid in changed_direct:
            trace.append(TraceEntry(step, rid, trigger.get(rid, f"change:{rid}"), None))
        elif rid in new_parent:
            parent = new_parent[rid]
            trace.append(TraceEntry(step, rid, parent, f"{parent}->{rid}"))
        elif rid in old_parent:
            parent = old_parent[rid]
            if parent in deleted:
                trace.append(TraceEntry(step, rid, f"delete:{parent}",
                                        f"{parent}->>{rid}"))
            else:
                trace.append(TraceEntry(step, rid, parent, f"{parent}->{rid}"))
        else:  # 理论上不可达，防御性兜底
            trace.append(TraceEntry(step, rid, "change", None))
    return trace
