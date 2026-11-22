"""列组分层敏感性分析 (业务要求 5)."""

from __future__ import annotations

from collections import Counter

from .dequant import SOURCE_PRIORITY


def stratify_groups(
    tensor_name: str,
    group_labels: tuple[int, ...],
    column_results: dict[int, dict],
    quant_bits: int,
    group_oor_limit: float,
) -> dict:
    """按组标号聚合各列反量化结果.

    column_results: {column_index: {"dequant": ..., "quant": ..., "valid_count": ...}}
    返回按组标号排序的分层结论与超限 (组, 位宽) 组合列表.
    """
    groups: dict[int, list[int]] = {}
    for col, label in enumerate(group_labels):
        groups.setdefault(label, []).append(col)

    group_entries = []
    exceedances = []
    for label in sorted(groups):
        # 跳过失败/缺失列 (如预算耗尽), 仅统计有结果的列
        cols = sorted(c for c in groups[label] if c in column_results)
        max_rel = 0.0
        total_elements = 0
        clipped_elements = 0
        oor_elements = 0
        source_counter: Counter = Counter()
        for col in cols:
            res = column_results[col]
            dq = res["dequant"]
            qt = res["quant"]
            valid = res["valid_count"]
            total_elements += valid
            clipped_elements += qt["clipped_count"]
            oor_elements += dq["oor_count"]
            max_rel = max(max_rel, dq["max_rel_err"])
            for att in dq["attributions"]:
                if att is not None:
                    source_counter[att["source"]] += 1
        sat_ratio = clipped_elements / total_elements if total_elements else 0.0
        oor_ratio = oor_elements / total_elements if total_elements else 0.0
        total_attr = sum(source_counter.values())
        if total_attr:
            dominant, dominant_count = max(
                ((s, source_counter.get(s, 0)) for s in SOURCE_PRIORITY),
                key=lambda kv: (kv[1], -SOURCE_PRIORITY.index(kv[0])),
            )
            dominant_share = dominant_count / total_attr
        else:
            dominant, dominant_share = None, 0.0
        exceeded = oor_ratio > group_oor_limit
        if exceeded:
            exceedances.append({"tensor": tensor_name, "group": label,
                                "quant_bits": quant_bits,
                                "oor_ratio": oor_ratio, "limit": group_oor_limit})
        conclusion = (
            f"group={label} cols={len(cols)} max_rel_err={max_rel:.6g} "
            f"sat_ratio={sat_ratio:.4f} oor_ratio={oor_ratio:.4f} "
            f"dominant={dominant or 'none'}({dominant_share:.2f}) "
            f"{'EXCEEDS_LIMIT' if exceeded else 'within_limit'}"
        )
        group_entries.append(
            {
                "group": label,
                "columns": cols,
                "max_rel_err": max_rel,
                "saturation_ratio": sat_ratio,
                "oor_ratio": oor_ratio,
                "dominant_source": dominant,
                "dominant_share": dominant_share,
                "exceeds_limit": exceeded,
                "conclusion": conclusion,
            }
        )
    return {"tensor": tensor_name, "quant_bits": quant_bits,
            "groups": group_entries, "exceedances": exceedances}
