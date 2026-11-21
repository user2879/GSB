"""多来源证据按显式仲裁规则的合并判定（规则版本 arb-v1）。"""

from .models import METRICS

RULE_VERSION = "arb-v1"

# 每个指标的来源类型偏置（数值越小优先级越高），属仲裁规则的显式组成部分
SOURCE_TYPE_BIAS = {
    "capacity_limit_bytes": {"capacity_claim": 0, "derived_claim": 1, "rate_claim": 2},
    "ops_per_second": {"rate_claim": 0, "derived_claim": 1, "capacity_claim": 2},
    "burst_bucket_capacity": {"rate_claim": 0, "derived_claim": 1, "capacity_claim": 2},
}


def _reject_reason(metric, winner, loser):
    if SOURCE_TYPE_BIAS[metric][winner.source_type] != SOURCE_TYPE_BIAS[metric][loser.source_type]:
        return "source_type_bias"
    if winner.format_version != loser.format_version:
        return "older_format_version"
    return "source_id_order"


def arbitrate_volume(volume_id, records, rule_version=RULE_VERSION):
    """对单卷逐指标裁决。排序键 (类型偏置, -格式版本, 来源标识) 保证结论
    与记录读取顺序、字典遍历顺序无关。"""
    conclusions = {}
    for metric in METRICS:
        candidates = [r for r in records if metric in r.metrics]
        if not candidates:
            continue
        ordered = sorted(
            candidates,
            key=lambda r: (SOURCE_TYPE_BIAS[metric][r.source_type],
                           -r.format_version, r.source_id))
        winner = ordered[0]
        rejected = [{
            "source_id": r.source_id,
            "source_type": r.source_type,
            "format_version": r.format_version,
            "value": r.metrics[metric],
            "reason": _reject_reason(metric, winner, r),
        } for r in ordered[1:]]
        conclusions[metric] = {
            "volume_id": volume_id,
            "metric": metric,
            "value": winner.metrics[metric],
            "error_bound": winner.quantization_error,
            "adopted_source_id": winner.source_id,
            "adopted_source_type": winner.source_type,
            "adopted_format_version": winner.format_version,
            "rule_version": rule_version,
            "rejected": rejected,
        }
    return conclusions
