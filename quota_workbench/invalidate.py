"""来源记录或仲裁规则变更后的失效传播与结论整体替换。"""

from .models import canonical

# 来源类型 -> 该类型可提供的指标（派生声明按 fields.metric 判定）
_TYPE_METRICS = {
    "capacity_claim": {"capacity_limit_bytes"},
    "rate_claim": {"ops_per_second", "burst_bucket_capacity"},
}


def source_metrics(rec):
    if rec["source_type"] == "derived_claim":
        return {rec["fields"]["metric"]}
    return set(_TYPE_METRICS[rec["source_type"]])


def changed_source_ids(old_sources, new_sources):
    """按规范化 JSON 逐字节比较，找出新增、删除或修改的来源标识。"""
    changed = set()
    for sid in set(old_sources) | set(new_sources):
        old = old_sources.get(sid)
        new = new_sources.get(sid)
        if old is None or new is None or canonical(old) != canonical(new):
            changed.add(sid)
    return changed


def affected_domains(old_sources, new_sources, volumes, rule_changed=False):
    """沿卷 -> 来源 -> 指标引用关系推导受影响的 (volume_id, metric) 域。
    仲裁规则版本变更时全部域失效。"""
    if rule_changed:
        return {(vid, m) for vid in volumes
                for m in ("capacity_limit_bytes", "ops_per_second",
                          "burst_bucket_capacity")}
    changed = changed_source_ids(old_sources, new_sources)
    affected = set()
    for vid, sids in volumes.items():
        for sid in sids:
            if sid not in changed:
                continue
            rec = new_sources.get(sid) or old_sources.get(sid)
            for metric in source_metrics(rec):
                affected.add((vid, metric))
    return affected


def replace_view(old_rows, new_rows, affected):
    """结论视图整体替换：受影响域作废旧行、采纳新行，未受影响行逐字节保留。
    返回 (view, retired, added, preserved_intact)。"""
    def key(r):
        return (r["volume_id"], r["metric"])

    retired = [r for r in old_rows if key(r) in affected]
    kept = [r for r in old_rows if key(r) not in affected]
    added = [r for r in new_rows if key(r) in affected]
    # 校验未受影响结论逐字节不变
    new_kept = {key(r): r for r in new_rows if key(r) not in affected}
    preserved_intact = all(
        canonical(r) == canonical(new_kept[key(r)])
        for r in kept if key(r) in new_kept)
    view = sorted(kept + added,
                  key=lambda r: (r["volume_id"], r["metric"], canonical(r)))
    return view, retired, added, preserved_intact
