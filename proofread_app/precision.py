"""逐级误差传播追踪：配置精度档位 vs 高一档（float64）。"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import AppConfig
from .models import Candidate, PrecisionAudit, ScoreBreakdown, StageDiff

# 四个受追踪环节（顺序即传播链路顺序）
STAGE_DETECTION = "detection"
STAGE_SCORING = "item_scoring"
STAGE_AGGREGATION = "aggregation"
STAGE_DECISION = "decision"
STAGES = (STAGE_DETECTION, STAGE_SCORING, STAGE_AGGREGATION, STAGE_DECISION)


def _diff_pair(stage: str, low: Optional[float], high: Optional[float],
               tolerance: float) -> StageDiff:
    if low is None or high is None:
        abs_diff = 0.0
        rel_diff = 0.0
    else:
        # 偏差统一在 float64 下计算
        low64 = float(low)
        high64 = float(high)
        abs_diff = abs(high64 - low64)
        denom = max(abs(high64), 1e-12)
        rel_diff = abs_diff / denom
    return StageDiff(
        stage=stage,
        fp32_value=None if low is None else float(low),
        fp64_value=None if high is None else float(high),
        abs_diff=abs_diff,
        rel_diff=rel_diff,
        exceeds_tolerance=abs_diff > tolerance,
    )


def _score_index(scores: Sequence[ScoreBreakdown]) -> Dict[Tuple[str, int], ScoreBreakdown]:
    return {(item.rule_id, item.start): item for item in scores}


def audit_sample(sample_id: str, candidates: Sequence[Candidate],
                 scores_low: Sequence[ScoreBreakdown],
                 scores_high: Sequence[ScoreBreakdown],
                 config: AppConfig) -> List[PrecisionAudit]:
    low_map = _score_index(scores_low)
    high_map = _score_index(scores_high)
    audits: List[PrecisionAudit] = []
    keys = sorted(set(low_map) | set(high_map), key=lambda key: (key[1], key[0]))

    for rule_id, start in keys:
        low = low_map.get((rule_id, start))
        high = high_map.get((rule_id, start))
        candidate = next((c for c in candidates if c.rule_id == rule_id and c.start == start),
                         None)
        tolerance = float(config.tolerance)

        sim_low = candidate.similarity if candidate and config.precision == "float32" else (
            candidate.similarity if candidate else None)
        # 相似度本身由集合运算得到，两档位输入一致，差异为 0；仍走统一差值通道
        diffs = [
            _diff_pair(STAGE_DETECTION,
                       candidate.similarity if candidate else None,
                       candidate.similarity if candidate else None,
                       tolerance),
            _diff_pair(STAGE_SCORING,
                       low.detect_raw if low else None,
                       high.detect_raw if high else None,
                       tolerance),
            _diff_pair(STAGE_AGGREGATION,
                       low.total_confidence if low else None,
                       high.total_confidence if high else None,
                       tolerance),
        ]

        decision_flip = bool(low and high and low.decision != high.decision)
        if decision_flip:
            decision_diff = StageDiff(
                stage=STAGE_DECISION,
                fp32_value=0.0 if low and low.decision == "apply" else 1.0,
                fp64_value=0.0 if high and high.decision == "apply" else 1.0,
                abs_diff=1.0,
                rel_diff=1.0,
                exceeds_tolerance=True,
            )
        else:
            decision_diff = StageDiff(STAGE_DECISION, None, None, 0.0, 0.0, False)
        diffs.append(decision_diff)

        # 分项误差与聚合误差排序矛盾：聚合误差不小于最大分项误差却出现决策翻转，
        # 或聚合误差严格小于某分项误差（量化抵消）时标记
        item_abs = diffs[1].abs_diff
        agg_abs = diffs[2].abs_diff
        ranking_contradiction = (
            not decision_flip and agg_abs + 1e-15 < item_abs and item_abs > 0.0
        ) or (decision_flip and agg_abs <= tolerance)

        first_breach = next((d.stage for d in diffs if d.exceeds_tolerance), None)
        max_abs = max(d.abs_diff for d in diffs)

        reasons = []
        if decision_flip:
            reasons.append("两档位编辑决策不一致：聚合总置信度跨阈值，差异来源于上游分项舍入传播")
        if ranking_contradiction:
            reasons.append("分项误差与聚合误差排序矛盾：定点量化抵消改变了误差量级")
        if not reasons:
            if first_breach:
                reasons.append(f"误差首次在 {first_breach} 环节超过容差 {tolerance:g}")
            else:
                reasons.append(f"全部环节差值均在容差 {tolerance:g} 内，误差未显著传播")

        audits.append(PrecisionAudit(
            sample_id=sample_id,
            rule_id=rule_id,
            start=start,
            stage_diffs=[d.to_fingerprint_obj() for d in diffs],
            first_breach_stage=first_breach,
            decision_flip=decision_flip,
            ranking_contradiction=ranking_contradiction,
            attribution="；".join(reasons),
            max_abs_diff=max_abs,
        ))
    audits.sort(key=lambda item: (item.start, item.rule_id))
    return audits


def format_sig6(value: float) -> float:
    """以 float64 保留至少 6 位有效数字输出（JSON 序列化时保留完整尾数）。"""
    if value == 0.0:
        return 0.0
    return float(f"{float(value):.9g}")
