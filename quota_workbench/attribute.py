"""不可迁移情形的证据级冲突归因。"""

from .arbitrate import arbitrate_volume

CAUSES = ("profile_difference", "format_version_difference", "default_fill_difference")


def _value(volume_id, records, metric, rule_version):
    concl = arbitrate_volume(volume_id, records, rule_version).get(metric)
    return (concl["value"], concl) if concl else (None, None)


def attribute_conflicts(volume_id, migration_rows, records, profile, rule_version):
    """对冲突指标做三步切换分解，贡献可复算：

    V0 = 当前口径结论
    V1 = 仅切换来源类型口径（profile_difference）
    V2 = 在 V1 基础上仅保留各类型最高格式版本（format_version_difference）
    V3 = 在 V2 基础上剔除该指标由默认补齐的记录（default_fill_difference）
    贡献 = 相邻中间值之差绝对值；残差 = 总偏差 - 三类贡献之和。
    输出按贡献降序、指标名升序的稳定规则排序。
    """
    allowed = set(profile["allowed_source_types"])
    min_ver = profile.get("min_format_version", 0)
    rows = []
    for mrow in migration_rows:
        if mrow["status"] != "conflict":
            continue
        metric = mrow["metric"]
        v0 = mrow["current_value"]
        step1 = [r for r in records if r.source_type in allowed]
        v1, _ = _value(volume_id, step1, metric, rule_version)
        best_ver = {}
        for r in step1:
            best_ver[r.source_type] = max(best_ver.get(r.source_type, 0),
                                          r.format_version)
        step2 = [r for r in step1
                 if r.format_version == best_ver[r.source_type]
                 and r.format_version >= min_ver]
        v2, _ = _value(volume_id, step2, metric, rule_version)
        step3 = [r for r in step2
                 if metric in r.metrics and metric not in r.defaulted_fields]
        v3, _ = _value(volume_id, step3, metric, rule_version)
        intermediates = {"v_current": v0, "v_step1": v1, "v_step2": v2,
                         "v_step3": v3, "target_value": mrow["target_value"]}
        pairs = [("profile_difference", v0, v1),
                 ("format_version_difference", v1, v2),
                 ("default_fill_difference", v2, v3)]
        contributions = []
        for cause, va, vb in pairs:
            contribution = abs(va - vb) if (va is not None and vb is not None) else None
            contributions.append((cause, contribution))
        known = sum(c for _, c in contributions if c is not None)
        residual = (mrow["deviation"] - known) if mrow["deviation"] is not None else None
        for cause, contribution in contributions:
            rows.append({
                "volume_id": volume_id,
                "metric": metric,
                "cause": cause,
                "contribution": contribution,
                "residual": residual,
                "intermediates": intermediates,
                "rule_version": rule_version,
                "profile_id": profile["profile_id"],
                "profile_version": profile["profile_version"],
            })
    # 稳定排序：贡献降序（None 排最后）、指标名升序、成因序固定
    cause_order = {c: i for i, c in enumerate(CAUSES)}
    rows.sort(key=lambda r: (-(r["contribution"] if r["contribution"] is not None else -1),
                             r["metric"], cause_order[r["cause"]]))
    return rows
