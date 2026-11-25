"""规范化连接图、候选计划枚举与确定性计划标识。

枚举规则：
- 仅枚举“连通序”：每一步新加入的块组必须与已加入集合存在等值连接谓词，
  因此被剪除的正是会产生笛卡尔积且无等值谓词的顺序；
- 每个连接步骤可独立选择 hash/sort_merge/nested_loop 三种算法；
- 计划标识是规范化算子序列的内容摘要，不随声明编辑顺序或枚举顺序变化；
- 计划集合在返回前按计划标识排序，保证逐字节一致。
"""

from itertools import product
from typing import Dict, FrozenSet, List, Sequence, Tuple

from .canonical import canonical_text, short_digest
from .errors import SpecError
from .model import JoinPredicate, QuerySpec

JOIN_ALGORITHMS = ("hash", "sort_merge", "nested_loop")


class JoinGraph:
    def __init__(self, query: QuerySpec):
        self.query = query
        self.nodes = sorted(query.bg_ids())
        self.adjacency: Dict[str, List[str]] = {node: [] for node in self.nodes}
        self.edge_predicates: Dict[FrozenSet[str], List[JoinPredicate]] = {}
        for pred in query.joins:
            edge = frozenset((pred.left_bg, pred.right_bg))
            self.edge_predicates.setdefault(edge, []).append(pred)
            self.adjacency[pred.left_bg].append(pred.right_bg)
            self.adjacency[pred.right_bg].append(pred.left_bg)
        for node in self.adjacency:
            self.adjacency[node] = sorted(set(self.adjacency[node]))
        self._validate_connected()

    def _validate_connected(self):
        if len(self.nodes) <= 1:
            return
        seen = {self.nodes[0]}
        stack = [self.nodes[0]]
        while stack:
            current = stack.pop()
            for neighbor in self.adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        missing = sorted(set(self.nodes) - seen)
        if missing:
            raise SpecError(
                "连接图不连通：存在无等值谓词路径的块组，将产生笛卡尔积",
                object_id=self.query.query_id,
                field=f"joins.{missing[0]}",
            )

    def predicates_between(self, joined: FrozenSet[str], new_node: str):
        result = []
        for neighbor in self.adjacency[new_node]:
            if neighbor in joined:
                result.extend(self.edge_predicates[frozenset((new_node, neighbor))])
        return sorted(
            result,
            key=lambda pred: (pred.left_bg, pred.left_col, pred.right_bg, pred.right_col),
        )


def enumerate_connected_orders(graph: JoinGraph) -> List[Tuple[str, ...]]:
    """枚举所有连通排列；起始节点按标识序，保证确定性。"""
    orders: List[Tuple[str, ...]] = []

    def dfs(prefix: Tuple[str, ...], used: FrozenSet[str]):
        if len(prefix) == len(graph.nodes):
            orders.append(prefix)
            return
        candidates = sorted(
            node
            for node in graph.nodes
            if node not in used and any(neighbor in used for neighbor in graph.adjacency[node])
        )
        for node in candidates:
            dfs(prefix + (node,), used | {node})

    for start in graph.nodes:
        dfs((start,), frozenset((start,)))
    return orders


def _scan_filters(query: QuerySpec, bg_id: str):
    return sorted(
        (pred for pred in query.filters if pred.bg_id == bg_id),
        key=lambda pred: (pred.column, pred.op, canonical_text(pred.value)),
    )


def _scan_step(query: QuerySpec, bg_id: str) -> Dict[str, object]:
    filters = _scan_filters(query, bg_id)
    return {
        "op": "scan",
        "bg": bg_id,
        "filters": [
            {"column": f.column, "op": f.op, "value": f.value} for f in filters
        ],
    }


def _join_step(
    bg_id: str, predicates: Sequence[JoinPredicate], algorithm: str
) -> Dict[str, object]:
    return {
        "op": "join",
        "algorithm": algorithm,
        "new_bg": bg_id,
        "predicates": [
            {
                "left_bg": p.left_bg,
                "left_col": p.left_col,
                "right_bg": p.right_bg,
                "right_col": p.right_col,
            }
            for p in predicates
        ],
    }


def build_canonical_plan(query: QuerySpec, order: Tuple[str, ...], algorithms: Tuple[str, ...]):
    graph = JoinGraph(query)
    steps: List[Dict[str, object]] = []
    used = frozenset((order[0],))
    steps.append(_scan_step(query, order[0]))
    join_index = 0
    for bg_id in order[1:]:
        predicates = graph.predicates_between(used, bg_id)
        steps.append(_scan_step(query, bg_id))
        steps.append(_join_step(bg_id, predicates, algorithms[join_index]))
        used = used | frozenset((bg_id,))
        join_index += 1
    steps.append(
        {
            "op": "aggregate",
            "group_by": sorted(
                (
                    {"bg": ref["bg"], "column": ref["column"]}
                    for ref in query.group_by
                ),
                key=lambda item: (item["bg"], item["column"]),
            ),
            "aggregates": sorted(
                (
                    {
                        "func": agg.func,
                        "bg": agg.bg_id,
                        "column": agg.column,
                        "alias": agg.alias,
                    }
                    for agg in query.aggregates
                ),
                key=lambda item: (item["alias"], item["func"], item["bg"]),
            ),
        }
    )
    body = {
        "query_id": query.query_id,
        "order": list(order),
        "steps": steps,
    }
    plan_id = "P" + short_digest(body, 16)
    return {"plan_id": plan_id, "body": body}


def enumerate_plans(query: QuerySpec) -> List[Dict[str, object]]:
    graph = JoinGraph(query)
    orders = enumerate_connected_orders(graph)
    join_count = len(graph.nodes) - 1
    algorithm_combos = (
        list(product(JOIN_ALGORITHMS, repeat=join_count)) if join_count else [()]
    )
    plans = []
    seen_ids = set()
    for order in orders:
        for algorithms in algorithm_combos:
            plan = build_canonical_plan(query, order, algorithms)
            if plan["plan_id"] not in seen_ids:
                seen_ids.add(plan["plan_id"])
                plans.append(plan)
    plans.sort(key=lambda plan: plan["plan_id"])
    return plans
