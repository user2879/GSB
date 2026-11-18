"""规则依赖图。

边的语义：upstream -> downstream（上游规则被下游规则引用）。
失效传播沿 downstream 方向进行；拓扑排序保证下游总在上游之后。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .errors import DependError, FieldViolation
from .models import Rule


@dataclass(frozen=True)
class TraceEntry:
    step: int
    rule_id: str          # 本步被失效的规则
    triggered_by: str     # 触发源：规则标识或 "change:<id>" / "delete:<id>"
    via_edge: str | None  # 经过的依赖边 "upstream->downstream"，直接变更时为 None

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "rule_id": self.rule_id,
            "triggered_by": self.triggered_by,
            "via_edge": self.via_edge,
        }


class DependencyGraph:
    def __init__(self, rules: Iterable[Rule]):
        self.rules: dict[str, Rule] = {r.rule_id: r for r in rules}
        # downstream[u] = 直接引用 u 的规则集合
        self.downstream: dict[str, set[str]] = {rid: set() for rid in self.rules}
        self.upstream: dict[str, tuple[str, ...]] = {
            rid: r.upstream for rid, r in self.rules.items()
        }
        for rule in self.rules.values():
            for up in rule.upstream:
                # 未知上游由 validate_references 报告；构造期容错避免 KeyError。
                self.downstream.setdefault(up, set()).add(rule.rule_id)

    # ---- 校验 -----------------------------------------------------------
    def validate_references(self) -> list[FieldViolation]:
        violations: list[FieldViolation] = []
        for rule in self.rules.values():
            for up in rule.upstream:
                if up not in self.rules:
                    violations.append(FieldViolation(
                        rule.rule_id, "upstream",
                        f"上游规则标识 {up!r} 不存在"))
                if up == rule.rule_id:
                    violations.append(FieldViolation(
                        rule.rule_id, "upstream", "规则不能引用自身"))
        return violations

    def ensure_acyclic(self) -> None:
        """ Kahn 算法检测成环，失败抛 E_DEPEND 并列出环上节点。"""
        indegree = {rid: len(self.upstream[rid]) for rid in self.rules}
        ready = sorted(rid for rid, deg in indegree.items() if deg == 0)
        visited: list[str] = []
        while ready:
            node = ready.pop(0)
            visited.append(node)
            for child in sorted(self.downstream[node]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready.sort()
        if len(visited) != len(self.rules):
            cycle_nodes = sorted(set(self.rules) - set(visited))
            raise DependError(
                "依赖图存在环，无法编译",
                details={"cycle_nodes": cycle_nodes},
            )

    def validate_all(self) -> list[FieldViolation]:
        violations = self.validate_references()
        if not violations:
            self.ensure_acyclic()
        return violations

    # ---- 拓扑 -----------------------------------------------------------
    def topo_order(self) -> list[str]:
        """稳定拓扑序：同层按规则标识字典序，结果完全确定。"""
        indegree = {rid: len(self.upstream[rid]) for rid in self.rules}
        ready = sorted(rid for rid, deg in indegree.items() if deg == 0)
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for child in sorted(self.downstream[node]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready.sort()
        if len(order) != len(self.rules):
            raise DependError("依赖图存在环，无法拓扑排序",
                              details={"remaining": sorted(set(self.rules) - set(order))})
        return order

    # ---- 传播 -----------------------------------------------------------
    def transitive_downstream(self, roots: Iterable[str]) -> set[str]:
        """返回 roots 的全部直接与间接下游（不含 roots 自身）。"""
        affected: set[str] = set()
        stack = list(roots)
        while stack:
            node = stack.pop()
            for child in self.downstream.get(node, ()):  # type: ignore[arg-type]
                if child not in affected:
                    affected.add(child)
                    stack.append(child)
        return affected

    def propagate_invalidation(
        self, changed: set[str], trigger_source: dict[str, str] | None = None,
    ) -> tuple[list[str], list[TraceEntry]]:
        """按稳定拓扑顺序推导失效序列，并记录每步触发源。

        changed: 直接被变更（新增/修改）的规则标识集合。
        trigger_source: 规则标识 -> 触发源描述（如 change:R1 / delete:R2）。
        返回 (失效序列, 传播轨迹)。失效序列按拓扑序排列，含 changed 本身。
        """
        trigger_source = trigger_source or {rid: f"change:{rid}" for rid in changed}
        affected_extras = self.transitive_downstream(changed)
        invalid_set = set(changed) | affected_extras
        # 失效序列遵循全图稳定拓扑序，保证下游在其上游之后失效。
        sequence = [rid for rid in self.topo_order() if rid in invalid_set]

        # 重新构造每条边的最短触发链：BFS 记录直接父来源。
        parent_of: dict[str, str] = {}
        frontier = list(changed)
        seen = set(changed)
        while frontier:
            node = frontier.pop(0)
            for child in sorted(self.downstream.get(node, ())):  # type: ignore[arg-type]
                if child not in seen:
                    seen.add(child)
                    parent_of[child] = node
                    frontier.append(child)

        trace: list[TraceEntry] = []
        for step, rid in enumerate(sequence, start=1):
            if rid in changed:
                trace.append(TraceEntry(step, rid, trigger_source.get(rid, f"change:{rid}"),
                                        None))
            else:
                parent = parent_of[rid]
                trace.append(TraceEntry(step, rid, parent, f"{parent}->{rid}"))
        return sequence, trace
