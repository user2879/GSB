"""Stratified attribution with weighted rollup and drift guard.

For every bucketing attribute key, users are split into strata (attribute values).
Each stratum produces a primary-metric difference and confidence interval between
the control and each treatment. Strata are weighted by their pooled exposure
share and the per-stratum lift is multiplied by that weight to produce an
attribution share. The shares must reconstruct the overall absolute lift within
the configured fixed-point tolerance; otherwise E_ATTRIBUTION_DRIFT is raised and
a deterministic largest-share correction is emitted.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .config import ExperimentConfig
from .errors import E_ATTRIBUTION_DRIFT, FailureRecord
from .fingerprint import stable_sort_key
from .fixedpoint import FixedPoint
from .loader import Exposure
from .stats import CLICK_RATE, CONVERSION_RATE

_FLAG = {CLICK_RATE: "clicked", CONVERSION_RATE: "converted"}


def _stratum_of(user_id: str, attributes: Dict[str, Dict[str, str]], key: str) -> str:
    attrs = attributes.get(user_id)
    if not attrs or key not in attrs:
        return "__unattributed__"
    return attrs[key]


def _counts_in_stratum(
    exposures: List[Exposure],
    group: str,
    key: str,
    value: str,
    attributes: Dict[str, Dict[str, str]],
    metric: str,
) -> Tuple[int, int]:
    flag_name = _FLAG[metric]
    numerator = 0
    denominator = 0
    for exposure in exposures:
        if exposure.group != group:
            continue
        if _stratum_of(exposure.user_id, attributes, key) != value:
            continue
        denominator += 1
        numerator += getattr(exposure, flag_name)
    return numerator, denominator


def _ci_for_counts(c_num, c_den, t_num, t_den, fp: FixedPoint, z: float):
    p_c = c_num / c_den
    p_t = t_num / t_den
    diff = p_t - p_c
    se = math.sqrt(
        max(p_t * (1 - p_t) / t_den, 0.0) + max(p_c * (1 - p_c) / c_den, 0.0)
    )
    return (
        fp.floor_scaled(diff - z * se),
        fp.ceil_scaled(diff + z * se),
    )


def _largest_remainder_weights(weight_dens: List[int], scale: int) -> List[int]:
    """Allocate exactly ``scale`` units across strata by largest remainder."""
    total = sum(weight_dens)
    if total == 0:
        return [0 for _ in weight_dens]
    exact = [weight * scale / total for weight in weight_dens]
    floors = [int(x) for x in exact]
    remainder = scale - sum(floors)
    order = sorted(
        range(len(weight_dens)),
        key=lambda idx: (-(exact[idx] - floors[idx]), str(idx)),
    )
    for idx in order[:remainder]:
        floors[idx] += 1
    return floors


def _share(lift_scaled: Optional[int], weight_scaled: int, fp: FixedPoint) -> int:
    if lift_scaled is None:
        return 0
    return fp.quantize((lift_scaled * weight_scaled) // fp.scale)


def _attribute_key(
    key: str,
    exposures: List[Exposure],
    config: ExperimentConfig,
    attributes: Dict[str, Dict[str, str]],
    fp: FixedPoint,
) -> Tuple[Dict[str, Any], Optional[FailureRecord]]:
    metric = config.primary_metric
    values = sorted(
        {
            _stratum_of(e.user_id, attributes, key)
            for e in exposures
        }
    )
    rows: List[Dict[str, Any]] = []
    drift: Optional[FailureRecord] = None
    total_exposures = 0
    for treatment_group in config.treatment_groups:
        strata_rows: List[Dict[str, Any]] = []
        pooled = 0
        for value in values:
            c_num, c_den = _counts_in_stratum(
                exposures, config.control_group, key, value, attributes, metric
            )
            t_num, t_den = _counts_in_stratum(
                exposures, treatment_group, key, value, attributes, metric
            )
            weight_den = c_den + t_den
            pooled += weight_den
            if c_den == 0 or t_den == 0:
                lift_scaled = None
                lower = upper = None
            else:
                lift_scaled = fp.quantize(
                    fp.ratio(t_num, t_den) - fp.ratio(c_num, c_den)
                )
                lower, upper = _ci_for_counts(
                    c_num, c_den, t_num, t_den, fp, config.z_quantile
                )
            strata_rows.append(
                {
                    "stratum": value,
                    "control_n": c_num,
                    "control_d": c_den,
                    "treatment_n": t_num,
                    "treatment_d": t_den,
                    "weight_den": weight_den,
                    "lift_scaled": lift_scaled,
                    "ci_lower_scaled": lower,
                    "ci_upper_scaled": upper,
                }
            )
        total_exposures = pooled
        weights = _largest_remainder_weights(
            [row["weight_den"] for row in strata_rows], fp.scale
        )
        for row, weight_scaled in zip(strata_rows, weights):
            row["weight_scaled"] = weight_scaled
            row["weight_text"] = fp.to_text(weight_scaled)
            share = _share(row["lift_scaled"], weight_scaled, fp)
            row["share_scaled"] = share
            row["share_text"] = fp.to_text(share)
            if row["lift_scaled"] is None:
                row["lift_text"] = None
                row["ci_lower_text"] = None
                row["ci_upper_text"] = None
            else:
                row["lift_text"] = fp.to_text(row["lift_scaled"])
                row["ci_lower_text"] = fp.to_text(row["ci_lower_scaled"])
                row["ci_upper_text"] = fp.to_text(row["ci_upper_scaled"])
            row.pop("weight_den", None)

        rollup = sum(row["share_scaled"] for row in strata_rows)
        overall = _overall_lift(
            exposures, config.control_group, treatment_group, metric, fp
        )
        tolerance = _tolerance_scaled(config, fp)
        discrepancy = rollup - overall
        correction: List[Dict[str, Any]] = []
        if abs(discrepancy) > tolerance:
            drift = FailureRecord(
                stage="attribution",
                code=E_ATTRIBUTION_DRIFT,
                reason="weighted_rollup_drift",
                group=treatment_group,
                detail=f"key={key},discrepancy={discrepancy},tolerance={tolerance}",
            )
            correction = _correct(
                strata_rows, discrepancy, fp, lift_scaled_overall=overall
            )
            rollup = sum(row["share_scaled"] for row in strata_rows)
            discrepancy = rollup - overall
        rows.append(
            {
                "treatment": treatment_group,
                "control": config.control_group,
                "strata": strata_rows,
                "rollup_lift_scaled": rollup,
                "rollup_lift_text": fp.to_text(rollup),
                "overall_lift_scaled": overall,
                "overall_lift_text": fp.to_text(overall),
                "discrepancy_scaled": discrepancy,
                "tolerance_scaled": tolerance,
                "correction": correction,
            }
        )
    return {"attribute_key": key, "results": rows}, drift


def _overall_lift(
    exposures: List[Exposure],
    control: str,
    treatment: str,
    metric: str,
    fp: FixedPoint,
) -> int:
    flag_name = _FLAG[metric]
    c_num = c_den = t_num = t_den = 0
    for exposure in exposures:
        if exposure.group == control:
            c_den += 1
            c_num += getattr(exposure, flag_name)
        elif exposure.group == treatment:
            t_den += 1
            t_num += getattr(exposure, flag_name)
    if c_den == 0 or t_den == 0:
        return 0
    return fp.quantize(fp.ratio(t_num, t_den) - fp.ratio(c_num, c_den))


def _tolerance_scaled(config: ExperimentConfig, fp: FixedPoint) -> int:
    return max(
        1,
        int((Decimal(config.attribution_tolerance) * Decimal(fp.scale)).to_integral_value()),
    )


def _correct(
    strata_rows: List[Dict[str, Any]],
    discrepancy: int,
    fp: FixedPoint,
    lift_scaled_overall: int,
) -> List[Dict[str, Any]]:
    adjustable = [r for r in strata_rows if r["lift_scaled"] is not None]
    if not adjustable:
        return []
    corrections: List[Dict[str, Any]] = []
    remaining = -discrepancy
    if remaining > 0:
        candidates = sorted(
            adjustable,
            key=lambda r: (
                -r["weight_scaled"],
                -_remainder(r["lift_scaled"], r["weight_scaled"], fp.scale),
                r["stratum"],
            ),
        )
    else:
        candidates = sorted(
            adjustable,
            key=lambda r: (
                -r["weight_scaled"],
                r["stratum"],
            ),
        )
    sign = 1 if remaining > 0 else -1
    total_weight = sum(r["weight_scaled"] for r in candidates)
    cursor = 0
    while remaining != 0 and candidates:
        row = candidates[cursor % len(candidates)]
        if cursor < len(candidates):
            capacity = abs(remaining) * row["weight_scaled"] // max(total_weight, 1)
            if cursor == len(candidates) - 1:
                capacity = abs(remaining)
        else:
            capacity = abs(remaining)
        delta = sign * max(capacity, 1)
        if abs(delta) > abs(remaining):
            delta = sign * abs(remaining)
        before = row["share_scaled"]
        row["share_scaled"] = before + delta
        row["share_text"] = fp.to_text(row["share_scaled"])
        row["corrected"] = True
        corrections.append(
            {
                "stratum": row["stratum"],
                "before_share_scaled": before,
                "after_share_scaled": row["share_scaled"],
                "applied_delta_scaled": delta,
            }
        )
        remaining -= delta
        cursor += 1
    return corrections


def _remainder(lift_scaled: int, weight_scaled: int, scale: int) -> int:
    return abs(lift_scaled * weight_scaled) % scale


def attribute(
    exposures: List[Exposure],
    config: ExperimentConfig,
    attributes: Dict[str, Dict[str, str]],
    fp: FixedPoint,
    concurrent: bool = True,
) -> Tuple[List[Dict[str, Any]], List[FailureRecord]]:
    keys = sorted({k for attrs in attributes.values() for k in attrs})
    ordered = sorted(
        ((stable_sort_key("attribution-key", key), key) for key in keys),
        key=lambda t: t[0],
    )
    drifts: List[FailureRecord] = []
    output: List[Dict[str, Any]] = []

    if concurrent and config.concurrency > 1 and ordered:
        workers = min(config.concurrency, len(ordered))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(
                pool.map(
                    lambda item: _attribute_key(
                        item[1], exposures, config, attributes, fp
                    ),
                    ordered,
                )
            )
    else:
        results = [
            _attribute_key(key, exposures, config, attributes, fp)
            for _sort, key in ordered
        ]

    for row, drift in results:
        output.append(row)
        if drift is not None:
            drifts.append(drift)
    drifts.sort(key=lambda f: f.sort_key())
    return output, drifts
