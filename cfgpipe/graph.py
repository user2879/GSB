"""派生依赖图与变更集驱动的失效传播。

确定性保证：所有输出按稳定拓扑序（Kahn + 堆排序键）与字典序生成，
不依赖 dict/set 的遍历顺序；同一变更集在任意声明条目顺序下输出逐字节一致。
"""

import heapq
import json
from dataclasses import dataclass


def topo_order(item_keys, derived_from):
    """稳定拓扑序：父项先于派生项；同层按键名字典序。"""
    keys = sorted(item_keys)
    indeg = {k: 0 for k in keys}
    children = {k: [] for k in keys}
    for k in keys:
        parent = derived_from.get(k, "")
        if parent:
            indeg[k] += 1
            children[parent].append(k)
    heap = [k for k in keys if indeg[k] == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        node = heapq.heappop(heap)
        order.append(node)
        for child in sorted(children[node]):
            indeg[child] -= 1
            if indeg[child] == 0:
                heapq.heappush(heap, child)
    if len(order) != len(keys):
        raise ValueError("derived dependency graph has a cycle")
    return order


def dependents_map(derived_from):
    """key -> 排序后的直接派生方列表。"""
    deps = {}
    for key, parent in derived_from.items():
        if parent:
            deps.setdefault(parent, []).append(key)
    return {k: sorted(v) for k, v in deps.items()}


@dataclass(frozen=True)
class InvalidationStep:
    order: int
    kind: str      # "item" | "node"
    name: str
    reason: str


@dataclass(frozen=True)
class Invalidation:
    version: int
    items: tuple   # 受影响配置键，稳定拓扑序
    nodes: tuple   # 受影响节点，字典序
    steps: tuple   # InvalidationStep，按 order 升序

    def canonical(self):
        doc = {
            "version": self.version,
            "items": list(self.items),
            "nodes": list(self.nodes),
            "steps": [
                {"order": s.order, "kind": s.kind, "name": s.name, "reason": s.reason}
                for s in self.steps
            ],
        }
        return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _change_reasons(changeset):
    """内容变化（取值/过期/派生系数）导致的直接失效；订阅集合变化
    不使项失效（产物逐字节不变），仅影响新增/移除的订阅节点。"""
    reasons = {}
    for ch in changeset.changes:
        parts = []
        if ch.value is not None:
            parts.append("value")
        if ch.expire_ms is not None:
            parts.append("expire_ms")
        if ch.derived_coeff is not None:
            parts.append("derived_coeff")
        if parts:
            reasons[ch.key] = "changed in changeset v%d (%s)" % (
                changeset.version, ",".join(parts))
    return reasons


def compute_invalidation(state, changeset, node_subs):
    """推导失效集合。

    state: 应用变更集前的 WorldState（用于订阅差异对照）。
    changeset: 已校验的 Changeset。
    node_subs: 应用变更集后 node_id -> sorted订阅键 的映射。
    返回 Invalidation。
    """
    derived_from = {k: it.derived_from for k, it in state.items.items()}
    order_all = topo_order(list(state.items), derived_from)
    deps = dependents_map(derived_from)
    direct_reasons = _change_reasons(changeset)

    affected = set(direct_reasons)
    queue = list(sorted(direct_reasons))
    while queue:
        cur = queue.pop(0)
        for child in deps.get(cur, []):
            if child not in affected:
                affected.add(child)
                queue.append(child)

    item_reason = {}
    for key in sorted(direct_reasons):
        item_reason[key] = direct_reasons[key]
    for key in order_all:
        if key in affected and key not in item_reason:
            parent = derived_from.get(key, "")
            item_reason[key] = "derived_from '%s' invalidated" % parent

    affected_items = tuple(k for k in order_all if k in affected)

    # 节点：订阅了受影响项，或本轮订阅集合发生变化
    sub_changed = {}
    for ch in changeset.changes:
        for nid in ch.subscribers_add:
            sub_changed.setdefault(nid, []).append("subscribed to '%s'" % ch.key)
        for nid in ch.subscribers_remove:
            sub_changed.setdefault(nid, []).append("unsubscribed from '%s'" % ch.key)

    node_reasons = {}
    for nid in sorted(node_subs):
        reasons = []
        for key in node_subs[nid]:
            if key in affected:
                reasons.append("subscribes to invalidated '%s'" % key)
        # 被移除订阅的键也要计入（该键可能已不在 node_subs[nid] 中）
        reasons.extend(sorted(sub_changed.get(nid, [])))
        if reasons:
            node_reasons[nid] = sorted(set(reasons))

    steps = []
    idx = 0
    for key in affected_items:
        steps.append(InvalidationStep(idx, "item", key, item_reason[key]))
        idx += 1
    for nid in sorted(node_reasons):
        steps.append(InvalidationStep(
            idx, "node", nid, "; ".join(node_reasons[nid])))
        idx += 1

    return Invalidation(
        version=changeset.version,
        items=affected_items,
        nodes=tuple(sorted(node_reasons)),
        steps=tuple(steps),
    )


def node_subscription_map(state):
    """WorldState.subs (key -> nodes) 反转为 node_id -> sorted keys。"""
    out = {}
    for key, nids in state.subs.items():
        for nid in nids:
            out.setdefault(nid, []).append(key)
    return {nid: sorted(keys) for nid, keys in out.items()}
