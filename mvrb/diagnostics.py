"""跨版本一致性校验与边界诊断。"""
from __future__ import annotations

from collections import defaultdict

from .exchange_v1 import parse_v1, serialize_v1
from .exchange_v2 import parse_v2, serialize_v2
from .convert import downgrade_v2_to_v1, upgrade_v1_to_v2
from .ir import ReconstructionIR
from .metrics import METRIC_CODE_TO_NAME
from .utils import canonical_json

# 差异原因码（需求 6）
REASON_ROUNDING = "R_COORD_ROUNDING"
REASON_METRIC = "R_METRIC_MAPPING"
REASON_SIDECAR_MISS = "R_SIDECAR_MISS"

HIST_BINS = [0.0, 1e-9, 1e-6, 1e-3, 1e-1, 1.0, float("inf")]
HIST_LABELS = ["0", "<1e-9", "<1e-6", "<1e-3", "<1e-1", ">=1e-1"]


def _hist_bin(value: float) -> str:
    for i in range(len(HIST_BINS) - 1):
        if HIST_BINS[i] <= abs(value) < HIST_BINS[i + 1]:
            return HIST_LABELS[i]
    return HIST_LABELS[-1]


def round_trip_v1(ir: ReconstructionIR) -> ReconstructionIR:
    return parse_v1(serialize_v1(ir))


def round_trip_v2(ir: ReconstructionIR) -> ReconstructionIR:
    return parse_v2(serialize_v2(ir))


def cross_version_report(ir_v2: ReconstructionIR, sidecar: dict | None = None) -> dict:
    """同一逻辑结果分别 v1/v2 往返后重建，输出偏差与原因诊断。"""
    # v2 直接往返（无损）
    v2_back = round_trip_v2(ir_v2)
    # v2 -> v1（降级），再升版
    ir_v1, generated_sidecar = downgrade_v2_to_v1(ir_v2)
    v1_back = parse_v1(serialize_v1(ir_v1))
    upgraded = upgrade_v1_to_v2(v1_back, sidecar or generated_sidecar)

    report = {
        "per_view_coord_histogram": {},
        "max_score_abs_diff": 0.0,
        "metric_restore": [],
        "sidecar_hits": {"features": 0, "matches": 0, "voxel_points": 0,
                         "sidecar_supplied": sidecar is not None},
        "view_reasons": {},
    }

    # 逐视图坐标偏差直方图
    orig_feat = {(f.view, f.feature_id): f for f in ir_v2.features}
    hist = defaultdict(lambda: {label: 0 for label in HIST_LABELS})
    for f in upgraded.features:
        of = orig_feat.get((f.view, f.feature_id))
        if of is None:
            continue
        for diff in (f.x - of.x, f.y - of.y):
            hist[f.view][_hist_bin(diff)] += 1
    report["per_view_coord_histogram"] = {str(v): dict(h) for v, h in sorted(hist.items())}

    # 匹配得分最大偏差
    orig_match = {(m.view_a, m.feature_a, m.view_b, m.feature_b): m for m in ir_v2.matches}
    max_diff = 0.0
    for m in upgraded.matches:
        om = orig_match.get((m.view_a, m.feature_a, m.view_b, m.feature_b))
        if om is not None:
            max_diff = max(max_diff, abs(m.score_permille - om.score_permille))
    report["max_score_abs_diff"] = round(max_diff, 9)

    # 度量码还原情况
    for m in ir_v2.matches:
        restored_name = METRIC_CODE_TO_NAME.get(m.metric_code)
        report["metric_restore"].append({
            "view_a": m.view_a, "view_b": m.view_b,
            "metric": m.metric, "metric_code": m.metric_code,
            "restored_name": restored_name,
            "ok": restored_name == m.metric,
        })

    # 旁路命中
    sf = sidecar or generated_sidecar
    report["sidecar_hits"]["features"] = len(sf.get("features", {}))
    report["sidecar_hits"]["matches"] = len(sf.get("matches", {}))
    report["sidecar_hits"]["voxel_points"] = len(sf.get("voxel_points", {}))

    # 逐视图差异原因判定
    counts_orig = defaultdict(int)
    counts_up = defaultdict(int)
    for f in ir_v2.features:
        counts_orig[f.view] += 1
    for f in upgraded.features:
        counts_up[f.view] += 1
    views = sorted(set(counts_orig) | set(counts_up))
    for view in views:
        reasons = []
        if counts_orig[view] != counts_up[view]:
            reasons.append(REASON_ROUNDING)
        coord_diff = any(
            abs(f.x - orig_feat[(f.view, f.feature_id)].x) > 1e-9
            or abs(f.y - orig_feat[(f.view, f.feature_id)].y) > 1e-9
            for f in upgraded.features if (f.view, f.feature_id) in orig_feat
        )
        if coord_diff:
            reasons.append(REASON_ROUNDING)
        metric_bad = any(not item["ok"] for item in report["metric_restore"]
                         if item["view_a"] == view)
        if metric_bad:
            reasons.append(REASON_METRIC)
        # 显式提供了空旁路（旁路未命中）且存在坐标偏差
        sidecar_miss = (
            sidecar is not None
            and len(sidecar.get("features", {})) == 0
            and coord_diff
        )
        if sidecar_miss:
            reasons.append(REASON_SIDECAR_MISS)
        report["view_reasons"][str(view)] = sorted(set(reasons))

    # 三维点逐字段一致性（带旁路时必须完全一致）
    report["exact_restore_with_sidecar"] = (
        canonical_json(serialize_v2(upgraded)) == canonical_json(serialize_v2(ir_v2))
    )
    report["v2_lossless"] = (
        canonical_json(serialize_v2(v2_back)) == canonical_json(serialize_v2(ir_v2))
    )
    return report
