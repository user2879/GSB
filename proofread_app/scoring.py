"""定点化错误打分、加权聚合与编辑决策（float32 / float64 双档位）。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .config import AppConfig
from .models import (
    Candidate,
    FixedPointQuant,
    ScoreBreakdown,
    ScoringParams,
    RuleResource,
    ETYPE_PRIORITY,
)

DECISION_APPLY = "apply"
DECISION_SKIP = "skip"


@dataclass(frozen=True)
class Quantizer:
    """量化档位 + 定点小数位宽的确定性量化器。"""
    levels: int
    frac_bits: int
    precision: str

    @property
    def scale(self) -> float:
        return float(1 << self.frac_bits)

    def quantize(self, value: float) -> FixedPointQuant:
        target = float(value)
        # 先量化到有限档位，再定点化（两步都保留中间量）
        level_index = min(self.levels - 1,
                          max(0, int(round(target * (self.levels - 1)))))
        leveled = level_index / (self.levels - 1)
        quantized_int = int(round(leveled * self.scale))
        dequantized = quantized_int / self.scale
        return FixedPointQuant(
            raw=target,
            quantized_int=quantized_int,
            scale=self.scale,
            dequantized=dequantized,
            precision=self.precision,
        )


def _to_precision(value: float, precision: str):
    import numpy as np
    if precision == "float32":
        return float(np.float32(value))
    return float(np.float64(value))


def score_candidate(candidate: Candidate, resource: RuleResource,
                    params: ScoringParams, config: AppConfig,
                    precision: str) -> ScoreBreakdown:
    rule = resource.rule_map()[candidate.rule_id]
    etype = candidate.etype
    detect_weight = _to_precision(params.detect_weights[etype], precision)
    base_score = _to_precision(params.base_scores[etype], precision)
    rule_weight = _to_precision(rule.weight, precision)
    similarity = _to_precision(candidate.similarity, precision)
    coverage = _to_precision(candidate.length / max(1, len(candidate.window)), precision)

    # 检测分：基准分 × 检测权重 × 规则权重 × 相似度
    detect_raw = _to_precision(base_score * detect_weight * rule_weight * similarity,
                               precision)
    # 修正置信度：基准分 × 规则权重 ×（相似度与覆盖度的调和均值）
    # 修正置信度：基准分 × 规则权重 × 相似度与覆盖度的算术均值
    quality = _to_precision((similarity + coverage) / 2.0, precision)
    confidence_raw = _to_precision(base_score * rule_weight * quality, precision)

    quantizer = Quantizer(config.quant_levels, config.fixed_point_bits, precision)
    detect_quant = quantizer.quantize(detect_raw)
    confidence_quant = quantizer.quantize(confidence_raw)

    weights = config.aggregation_weights
    w_detect = _to_precision(weights["detect"], precision)
    w_conf = _to_precision(weights["confidence"], precision)
    weighted_detect = _to_precision(detect_quant.dequantized * w_detect, precision)
    weighted_confidence = _to_precision(confidence_quant.dequantized * w_conf, precision)
    total = _to_precision(weighted_detect + weighted_confidence, precision)
    threshold = _to_precision(config.decision_threshold, precision)

    if total > threshold:
        decision = DECISION_APPLY
        boundary = False
    elif total < threshold:
        decision = DECISION_SKIP
        boundary = False
    else:
        # 总置信度恰好等于阈值：按错误类型固定优先级决断
        boundary = True
        decision = DECISION_APPLY if ETYPE_PRIORITY[etype] <= 1 else DECISION_SKIP

    return ScoreBreakdown(
        sample_id=candidate.sample_id,
        rule_id=candidate.rule_id,
        etype=etype,
        start=candidate.start,
        end=candidate.end,
        detect_raw=detect_raw,
        confidence_raw=confidence_raw,
        detect_quant=detect_quant.to_fingerprint_obj(),
        confidence_quant=confidence_quant.to_fingerprint_obj(),
        weighted_detect=weighted_detect,
        weighted_confidence=weighted_confidence,
        total_confidence=total,
        decision=decision,
        boundary=boundary,
        precision=precision,
    )


def score_all(candidates: Sequence[Candidate], resource: RuleResource,
              params: ScoringParams, config: AppConfig, precision: str
              ) -> List[ScoreBreakdown]:
    breakdowns = [score_candidate(item, resource, params, config, precision)
                  for item in candidates]
    breakdowns.sort(key=lambda item: (item.start, item.rule_id))
    return breakdowns
