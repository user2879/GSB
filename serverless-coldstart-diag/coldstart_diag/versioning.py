"""声明版本演进、陈旧结论识别与当前视图整体替换（非按分钟增量合并）。"""

from .errors import SpecError
from .fixedpoint import indistinguishable

# 当前视图中参与跨版本比对的统计字段（稳定顺序）
VIEW_FIELDS = (
    "calls",
    "required_instances",
    "available_instances",
    "warm_target",
    "warm_instances",
    "reserved_used",
    "hot_calls",
    "cold_calls",
    "cold_starts",
    "queued_calls",
    "cold_ratio_q",
    "wait_ratio_q",
    "occupancy_q",
    "avg_wait_ms",
    "error_block",
)


def assert_monotonic(old_manifest, new_manifest):
    old_v, new_v = old_manifest["declaration_version"], new_manifest["declaration_version"]
    if new_v <= old_v:
        raise SpecError(
            "声明版本号必须单调递增",
            path="declaration_version",
            value={"old": old_v, "new": new_v},
        )


def diff_views(old_payloads, new_payloads):
    """整体替换时稳定列出发生变化的分钟、字段与旧新取值。"""
    old_map = {p["minute"]: p for p in old_payloads}
    new_map = {p["minute"]: p for p in new_payloads}
    changes = []
    for minute in sorted(set(old_map) | set(new_map)):
        o, n = old_map.get(minute), new_map.get(minute)
        if o is None or n is None:
            changes.append({"minute": minute, "field": "<row>", "old": o, "new": n})
            continue
        for field in VIEW_FIELDS:
            if o.get(field) != n.get(field):
                changes.append(
                    {"minute": minute, "field": field, "old": o.get(field), "new": n.get(field)}
                )
    return changes


def replace_view(old_payloads, new_payloads):
    """整体替换：返回新视图 + 变化清单；旧视图调用方负责标记 STALE。"""
    return {
        "replacement": "FULL_REPLACE",
        "stale_declaration": old_payloads[0]["declaration_version"] if old_payloads else None,
        "current_declaration": new_payloads[0]["declaration_version"] if new_payloads else None,
        "changed_minutes": sorted({c["minute"] for c in diff_views(old_payloads, new_payloads)}),
        "changes": diff_views(old_payloads, new_payloads),
        "current_view": new_payloads,
    }


def compare_version_stats(stat_a, stat_b):
    """两个版本聚合统计量之差落在合成误差界内时，按稳定规则判定不可区分。"""
    bound = max(stat_a["agg_error_bound_ulp"], stat_b["agg_error_bound_ulp"])
    verdicts = []
    for field in ("overall_cold_ratio_q", "overall_wait_ratio_q", "avg_occupancy_q"):
        a, b = stat_a[field], stat_b[field]
        diff = abs(a - b)
        decide = "INDISTINGUISHABLE" if diff <= bound else "DISTINGUISHABLE"
        # 平局稳定规则：差值相等时按字段名字典序固定输出，不依赖遍历顺序
        verdicts.append(
            {
                "field": field,
                "value_a": a,
                "value_b": b,
                "abs_diff": diff,
                "bound_ulp": bound,
                "verdict": decide,
                "tie_rule": "差值落界内判不可区分；并列时按字段名字典序稳定排序",
            }
        )
    return sorted(verdicts, key=lambda x: x["field"])
