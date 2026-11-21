"""节点侧（与权威重算侧共用）的整数定点分组聚合。

截断方向与溢出饱和规则：

* 重缩放与均值除法：向零截断（truncation toward zero）。
* 所有加/乘结果以有符号 64 位饱和（上溢 INT64_MAX、下溢 INT64_MIN），
  使用 Python 任意精度整数先行计算再饱和，杜绝 NumPy 静默回绕。
* 相同配置切片 + 相同样本声明 + 相同节点口径，输出规范化 JSON 字节
  逐字节一致（见 ``fixedpoint.canonical_bytes``）。
"""

from .fixedpoint import INT64_MAX, INT64_MIN, canonical_hex, sat_int64, trunc_div
from .samples import generate_samples, rescale_int


def _effective_raw(item, resolved):
    """计算配置项在其自身 scale 口径下的有效定点 raw（含派生系数）。"""
    if not item.derives_from:
        return item.value, item.scale
    total = 0
    for edge in item.derives_from:
        parent_raw, parent_scale = resolved[edge.key]
        # 先把父项重缩放至系数 scale，乘系数，再统一到当前项 scale。
        parent_at_coef = rescale_int(parent_raw, parent_scale, edge.coef_scale)
        product = sat_int64(parent_at_coef * edge.coef_raw)
        product_at_item = rescale_int(product, edge.coef_scale, item.scale)
        total = sat_int64(total + product_at_item)
    if item.value_type == "fixed":
        total = sat_int64(total + rescale_int(item.value, item.scale, item.scale))
    else:
        total = sat_int64(total + item.value)
    return total, item.scale


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
    return ordered


def aggregate_slice(node, version_number, slice_items, sample_decl):
    """执行分组聚合，返回 (summary_dict, canonical_hex)。

    summary 中的数值均为整数定点 raw，并显式携带 scale、截断与饱和声明。
    """
    groups, values = generate_samples(
        sample_decl.seed, sample_decl.n_samples, sample_decl.n_groups,
        sample_decl.value_range, sample_decl.declared_scale,
    )
    # 样本基准值重缩放至节点声明口径（向零截断）。
    base = [
        sat_int64(rescale_int(int(v), sample_decl.declared_scale, node.scale))
        for v in values
    ]

    items_by_key = {it.key: it for it in slice_items}
    if not items_by_key:
        raise ValueError("节点配置切片为空，无法执行聚合")
    resolved = {}
    for key in _topo_order(items_by_key):
        resolved[key] = _effective_raw(items_by_key[key], resolved)

    per_item = {}
    # 按键名稳定排序，避免任何字典顺序依赖。
    for key in sorted(resolved):
        item_raw, item_scale = resolved[key]
        effective = rescale_int(item_raw, item_scale, node.scale)
        accum = {}
        counts = {}
        for idx, g in enumerate(groups):
            g = int(g)
            product = sat_int64(base[idx] * effective)
            prior = accum.get(g, 0)
            accum[g] = sat_int64(prior + product)
            counts[g] = counts.get(g, 0) + 1
        group_summaries = {}
        for g in sorted(accum):
            total = sat_int64(accum[g])
            mean = trunc_div(total, counts[g])
            group_summaries[str(g)] = {
                "sum": {"raw": total, "scale": node.scale},
                "mean": {"raw": sat_int64(mean), "scale": node.scale},
                "count": counts[g],
            }
        per_item[key] = group_summaries

    summary = {
        "node_id": node.node_id,
        "config_version": version_number,
        "rules": {
            "truncation": "toward_zero",
            "overflow": "saturate_int64",
            "int64_min": INT64_MIN,
            "int64_max": INT64_MAX,
        },
        "group_key": node.group_key,
        "node_scale": node.scale,
        "items": per_item,
    }
    return summary, canonical_hex(summary)


def slice_digest(version_number, slice_items):
    """配置切片内容摘要：只依赖版本与切片项的规范化内容。"""
    payload = [
        {
            "key": it.key,
            "value_type": it.value_type,
            "value": it.value,
            "scale": it.scale,
            "targets": list(it.targets),
            "derives_from": [
                {"key": e.key, "coef_raw": e.coef_raw, "coef_scale": e.coef_scale}
                for e in it.derives_from
            ],
        }
        for it in sorted(slice_items, key=lambda s: s.key)
    ]
    return canonical_hex({"version": version_number, "slice": payload})
