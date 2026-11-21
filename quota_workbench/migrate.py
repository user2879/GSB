"""仲裁结论在另一来源口径上的迁移复核与可迁移性判定。"""

from .arbitrate import arbitrate_volume
from .models import METRICS


def migration_review(volume_id, current, records, profile, rule_version):
    """把当前生效配额迁移到目标来源口径重新复核。

    profile: {"profile_id", "profile_version", "allowed_source_types",
              "min_format_version"}
    偏差为 0 -> consistent；偏差 <= 合成误差界 -> indistinguishable；
    否则 -> conflict（输出所涉指标与偏差量）。
    """
    allowed = set(profile["allowed_source_types"])
    min_ver = profile.get("min_format_version", 0)
    filtered = [r for r in records
                if r.source_type in allowed and r.format_version >= min_ver]
    target = arbitrate_volume(volume_id, filtered, rule_version)
    rows = []
    for metric in METRICS:
        cur = current.get(metric)
        tgt = target.get(metric)
        if cur is None and tgt is None:
            continue
        if cur is None or tgt is None:
            rows.append({
                "volume_id": volume_id, "metric": metric, "status": "conflict",
                "deviation": None, "error_bound": None,
                "current_value": cur["value"] if cur else None,
                "target_value": tgt["value"] if tgt else None,
                "cause_hint": "metric_absent",
                "rule_version": rule_version,
                "profile_id": profile["profile_id"],
                "profile_version": profile["profile_version"],
            })
            continue
        deviation = abs(cur["value"] - tgt["value"])
        bound = cur["error_bound"] + tgt["error_bound"]
        if deviation == 0:
            status = "consistent"
        elif deviation <= bound:
            status = "indistinguishable"
        else:
            status = "conflict"
        rows.append({
            "volume_id": volume_id, "metric": metric, "status": status,
            "deviation": deviation, "error_bound": bound,
            "current_value": cur["value"], "target_value": tgt["value"],
            "rule_version": rule_version,
            "profile_id": profile["profile_id"],
            "profile_version": profile["profile_version"],
        })
    return rows
