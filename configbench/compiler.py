"""增量编译与派生依赖失效传播。

编译产物是规范化的不可变内容对象，其标识 = 内容 SHA256；因此未变化项
在跨版本时产物字节与标识逐字节不变，仅变化项得到新标识。
失效传播沿两条边求闭包：派生依赖图（下游项）与节点引用关系（同节点
上的兄弟项不传播，节点只是受影响落点）。所有遍历均按排序后的稳定顺序。
"""

from dataclasses import dataclass

from .fixedpoint import canonical_hex


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    version: int
    key: str
    content: dict
    targets: tuple


def _item_content(item):
    return {
        "key": item.key,
        "value_type": item.value_type,
        "value": item.value,
        "scale": item.scale,
        "targets": list(item.targets),
        "derives_from": [
            {"key": e.key, "coef_raw": e.coef_raw, "coef_scale": e.coef_scale}
            for e in item.derives_from
        ],
    }


def _topo_order(items):
    indegree = {k: 0 for k in items}
    dependents = {k: [] for k in items}
    for key, spec in items.items():
        for edge in spec.derives_from:
            indegree[key] += 1
            dependents[edge.key].append(key)
    ready = sorted(k for k, d in indegree.items() if d == 0)
    ordered = []
    while ready:
        cur = ready.pop(0)
        ordered.append(cur)
        for nxt in sorted(dependents[cur]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    return ordered, dependents


def _diff_items(prev_items, cur_items):
    """返回直接变化的配置项键（按排序顺序），变化口径=键值/目标集合/派生系数。"""
    changed = []
    for key in sorted(cur_items):
        cur = cur_items[key]
        if key not in prev_items:
            changed.append(key)
            continue
        prev = prev_items[key]
        prev_sig = (cur.value_type, cur.value, cur.scale, tuple(sorted(cur.targets)),
                    tuple(sorted((e.key, e.coef_raw, e.coef_scale)
                                 for e in cur.derives_from)))
        old_sig = (prev.value_type, prev.value, prev.scale, tuple(sorted(prev.targets)),
                   tuple(sorted((e.key, e.coef_raw, e.coef_scale)
                                for e in prev.derives_from)))
        if prev_sig != old_sig:
            changed.append(key)
    return changed


def propagate(prev_version, cur_version):
    """计算版本间失效传播。

    返回 dict:
      direct_changed: 直接变化键（稳定排序）
      invalidated: 全部直接+间接受影响键（稳定拓扑序）
      steps: 每步 {trigger_source, affected_keys, affected_nodes}
      affected_nodes: 全部受影响节点（稳定排序）
    """
    prev_items = prev_version.items if prev_version else {}
    cur_items = cur_version.items
    direct = _diff_items(prev_items, cur_items)
    topo, dependents = _topo_order(cur_items)

    invalidated = set(direct)
    steps = []
    frontier = list(direct)
    while frontier:
        source = frontier.pop(0)
        affected = sorted(k for k in dependents.get(source, [])
                          if k not in invalidated)
        if not affected:
            continue
        nodes = sorted({
            t for key in ([source] + affected)
            for t in cur_items[key].targets
        })
        steps.append({
            "trigger_source": source,
            "affected_keys": affected,
            "affected_nodes": nodes,
        })
        for key in affected:
            invalidated.add(key)
            frontier.append(key)
        frontier.sort()

    ordered_invalidated = [k for k in topo if k in invalidated]
    affected_nodes = sorted({
        t for key in ordered_invalidated for t in cur_items[key].targets
    })
    return {
        "direct_changed": direct,
        "invalidated": ordered_invalidated,
        "steps": steps,
        "affected_nodes": affected_nodes,
    }


def compile_version(prev_version, cur_version):
    """编译一个版本，返回 (artifacts: dict[key->Artifact], propagation)。"""
    propagation = propagate(prev_version, cur_version)
    artifacts = {}
    for key in sorted(cur_version.items):
        item = cur_version.items[key]
        content = _item_content(item)
        artifacts[key] = Artifact(
            artifact_id=canonical_hex(content),
            version=cur_version.version,
            key=key,
            content=content,
            targets=tuple(sorted(item.targets)),
        )
    return artifacts, propagation


def build_node_slices(version, registry):
    """把版本配置项按目标节点切分为 {node_id: [ItemSpec...]}。

    每个节点切片除直接目标项外，还闭包纳入其派生依赖项（节点聚合时
    需要父项才能计算派生值）；键与节点均稳定排序。
    """
    slices = {node_id: [] for node_id in sorted(registry.nodes)}

    def closure(item, acc):
        for edge in item.derives_from:
            parent = version.items[edge.key]
            if parent.key not in acc:
                acc.add(parent.key)
                closure(parent, acc)

    for key in sorted(version.items):
        item = version.items[key]
        needed = {key}
        closure(item, needed)
        for node_id in sorted(item.targets):
            for dep in sorted(needed):
                dep_item = version.items[dep]
                if dep_item not in slices[node_id]:
                    slices[node_id].append(dep_item)
    out = {}
    for node_id, items in slices.items():
        if items:
            unique = {it.key: it for it in items}
            out[node_id] = tuple(unique[k] for k in sorted(unique))
    return out
