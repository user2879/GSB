"""规范化序列化、内容摘要与确定性排序工具。

所有需要跨运行逐字节一致的产物（影响链、账本摘要、快照摘要）都经由
:func:`canonical_bytes` 序列化：键按字典序输出、浮点不参与规范化、整数
为十进制文本，从而消除 dict 遍历顺序差异。
"""
from __future__ import annotations

import hashlib
from collections import deque
from typing import Any, Hashable, Iterable


def canonical_bytes(obj: Any) -> bytes:
    """把 JSON 兼容对象编码为确定性字节串（sort_keys、无多余空白）。"""
    return _encode(obj).encode("utf-8")


def canonical_digest(obj: Any) -> str:
    """返回规范化对象的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def _encode(obj: Any) -> str:
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError("非有限浮点数不参与规范化序列化")
        return repr(obj)
    if isinstance(obj, str):
        return _json_string(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_encode(item) for item in obj) + "]"
    if isinstance(obj, dict):
        parts = []
        for key in sorted(obj):
            if not isinstance(key, str):
                raise TypeError("规范化映射的键必须是字符串")
            parts.append(_json_string(key) + ":" + _encode(obj[key]))
        return "{" + ",".join(parts) + "}"
    raise TypeError(f"不支持的规范化类型: {type(obj)!r}")


def _json_string(text: str) -> str:
    import json

    return json.dumps(text, ensure_ascii=False, separators=(",", ":"))


def topo_sort(nodes: Iterable[Hashable], edges: dict[Hashable, set[Hashable]]) -> list[Hashable]:
    """稳定 Kahn 拓扑排序。

    ``edges[u]`` 表示 ``u -> v`` 的下游集合；同层候选按字典序选取，
    因此对同一张图无论输入遍历顺序如何，输出都一致。
    """
    node_set = set(nodes)
    indegree = {node: 0 for node in node_set}
    for source, targets in edges.items():
        if source not in indegree:
            continue
        for target in targets:
            if target in indegree:
                indegree[target] += 1
    ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    ordered: list[Hashable] = []
    while ready:
        node = ready.popleft()
        ordered.append(node)
        for target in sorted(edges.get(node, ())):
            if target not in indegree:
                continue
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(ordered) != len(node_set):
        raise ValueError("图中存在环，无法拓扑排序")
    return ordered


def downstream_of(
    start: Hashable, edges: dict[Hashable, set[Hashable]]
) -> dict[Hashable, int]:
    """返回 ``start`` 的全部下游节点及其最短层级深度（深度 1 为直接下游）。"""
    depths: dict[Hashable, int] = {}
    queue: deque[tuple[Hashable, int]] = deque([(start, 0)])
    while queue:
        node, depth = queue.popleft()
        for nxt in sorted(edges.get(node, ())):
            if nxt == start or nxt in depths:
                continue
            depths[nxt] = depth + 1
            queue.append((nxt, depth + 1))
    return depths
