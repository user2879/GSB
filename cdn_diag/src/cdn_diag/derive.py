"""派生关系图：拓扑序、下游推导、确定性内容组合。"""

import hashlib

from .errors import GraphError

COMBINE_RULES = ("concat",)


def _children_map(contents):
    """path -> 直接下游路径集合（谁派生自它）。"""
    children = {p: set() for p in contents}
    for path, item in contents.items():
        for parent in item.derived_from:
            children[parent].add(path)
    return children


def topo_order(contents):
    """稳定拓扑序：Kahn 算法，候选集按路径字典序取，保证与字典遍历顺序无关。"""
    indegree = {p: len(item.derived_from) for p, item in contents.items()}
    ready = sorted(p for p, d in indegree.items() if d == 0)
    order = []
    children = _children_map(contents)
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
        ready.sort()
    if len(order) != len(contents):
        cyclic = sorted(p for p, d in indegree.items() if d > 0)
        raise GraphError(
            "派生关系存在环",
            first_path=cyclic[0],
            cyclic_paths=cyclic,
        )
    return order


def combine_bodies(rule, parent_items):
    """确定性组合规则：concat 按父路径字典序拼接字节。"""
    if rule == "concat":
        return b"\n".join(item.body for _, item in parent_items)


def combine_etag(rule, parent_items):
    """派生 ETag：父 (path, etag) 按路径排序后的摘要，确定性。"""
    if rule == "concat":
        material = ",".join(f"{p}={item.etag}" for p, item in parent_items)
        return "D" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def recompute_derived(contents):
    """按稳定拓扑序重算全部派生路径的 body 与 etag（就地更新）。"""
    for path in topo_order(contents):
        item = contents[path]
        if not item.derived_from:
            continue
        parents = sorted((p, contents[p]) for p in item.derived_from)
        item.body = combine_bodies(item.combine, parents)
        item.etag = combine_etag(item.combine, parents)


def downstream_of(contents, changed_paths):
    """给定变化路径集合，返回全部直接与间接下游路径（稳定拓扑序，不含自身）。"""
    children = _children_map(contents)
    affected = set()
    stack = list(changed_paths)
    while stack:
        node = stack.pop()
        for child in children[node]:
            if child not in affected:
                affected.add(child)
                stack.append(child)
    order = topo_order(contents)
    return [p for p in order if p in affected]
