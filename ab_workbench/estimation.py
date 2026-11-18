"""Point estimates, fixed-point confidence intervals and decision labels.

Variance estimator (two independent proportions, unpooled):
    var(p_t - p_c) = p_t(1-p_t)/n_t + p_c(1-p_c)/n_c
The fixed normal quantile is taken from the config (default z for 95%).
Absolute endpoints use conservative integer bounds (floor for lower, ceil for
upper) before quantization; relative lift divides by the control proportion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import ExperimentConfig
from .fixedpoint import FixedPoint
from .stats import GroupStat

DECISION_WIN = "win"
DECISION_TIE = "tie"
DECISION_LOSE = "lose"
DECISION_INSUFFICIENT = "insufficient_sample"


@dataclass
class Estimate:
    treatment: str
    control: str
    metric: str
    sufficient: bool
    boundary: str
    suppressed: bool
    absolute_lift_scaled: Optional[int]
    relative_lift_scaled: Optional[int]
    ci_lower_scaled: Optional[int]
    ci_upper_scaled: Optional[int]
    absolute_lift_text: Optional[str]
    relative_lift_text: Optional[str]
    ci_lower_text: Optional[str]
    ci_upper_text: Optional[str]
    control_value_scaled: int
    treatment_value_scaled: int
    decision: str

    def to_dict(self):
        return self.__dict__.copy()


def _counts(stat: GroupStat, metric: str):
    row = stat.metrics[metric]
    return row["numerator"], row["denominator"]


def _min_lift_scaled(config: ExperimentConfig, fp: FixedPoint) -> int:
    return int((Decimal(config.min_lift) * Decimal(fp.scale)).to_integral_value())


def estimate_pair(
    control: GroupStat,
    treatment: GroupStat,
    metric: str,
    config: ExperimentConfig,
    fp: FixedPoint,
) -> Estimate:
    c_num, c_den = _counts(control, metric)
    t_num, t_den = _counts(treatment, metric)
    threshold = config.min_sample_size
    sufficient_c = c_den >= threshold
    sufficient_t = t_den >= threshold
    sufficient = sufficient_c and sufficient_t

    def _boundary(den: int, ok: bool) -> str:
        if den == threshold:
            return "exactly_at_threshold:sufficient"
        return "above_threshold" if ok else "below_threshold:insufficient"

    boundary = (
        f"control({_boundary(c_den, sufficient_c)});"
        f"treatment({_boundary(t_den, sufficient_t)})"
    )

    base = Estimate(
        treatment=treatment.group,
        control=control.group,
        metric=metric,
        sufficient=sufficient,
        boundary=boundary,
        suppressed=not sufficient,
        absolute_lift_scaled=None,
        relative_lift_scaled=None,
        ci_lower_scaled=None,
        ci_upper_scaled=None,
        absolute_lift_text=None,
        relative_lift_text=None,
        ci_lower_text=None,
        ci_upper_text=None,
        control_value_scaled=control.metrics[metric]["value_scaled"],
        treatment_value_scaled=treatment.metrics[metric]["value_scaled"],
        decision=DECISION_INSUFFICIENT if not sufficient else DECISION_TIE,
    )
    if not sufficient or c_den == 0 or t_den == 0:
        return base

    p_c = c_num / c_den
    p_t = t_num / t_den
    diff = p_t - p_c
    variance = p_t * (1.0 - p_t) / t_den + p_c * (1.0 - p_c) / c_den
    se = math.sqrt(max(variance, 0.0))
    margin = config.z_quantile * se
    lower_float = diff - margin
    upper_float = diff + margin

    lower_scaled = fp.floor_scaled(lower_float)
    upper_scaled = fp.ceil_scaled(upper_float)
    abs_lift_scaled = fp.quantize(fp.ratio(t_num, t_den) - fp.ratio(c_num, c_den))

    if c_num == 0:
        rel_scaled = None
        rel_text = None
    else:
        rel_scaled = fp.quantize(
            (abs_lift_scaled * fp.scale) // max(control.metrics[metric]["value_scaled"], 1)
        )
        rel_text = fp.to_text(rel_scaled)

    min_lift = _min_lift_scaled(config, fp)
    decision = _decide(lower_scaled, upper_scaled, min_lift, config.metric_direction)

    base.absolute_lift_scaled = abs_lift_scaled
    base.relative_lift_scaled = rel_scaled
    base.ci_lower_scaled = lower_scaled
    base.ci_upper_scaled = upper_scaled
    base.absolute_lift_text = fp.to_text(abs_lift_scaled)
    base.relative_lift_text = rel_text
    base.ci_lower_text = fp.to_text(lower_scaled)
    base.ci_upper_text = fp.to_text(upper_scaled)
    base.decision = decision
    return base


def _decide(lower: int, upper: int, min_lift: int, direction: str) -> str:
    if direction == "higher_is_better":
        if lower > min_lift:
            return DECISION_WIN
        if upper < -min_lift:
            return DECISION_LOSE
        return DECISION_TIE
    if upper < -min_lift:
        return DECISION_WIN
    if lower > min_lift:
        return DECISION_LOSE
    return DECISION_TIE


def estimate_all(
    group_stats: Dict[str, GroupStat], config: ExperimentConfig, fp: FixedPoint
) -> List[Estimate]:
    control = group_stats[config.control_group]
    estimates: List[Estimate] = []
    for treatment_group in config.treatment_groups:
        treatment = group_stats[treatment_group]
        estimates.append(
            estimate_pair(control, treatment, config.primary_metric, config, fp)
        )
    return estimates
