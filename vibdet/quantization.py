"""Normalization, fixed-point quantization, fusion and threshold verdicts.

Pipeline conventions:
* detector scores are normalized to [0, 1];
* each normalized score is converted to a fixed-point integer with
  ``2 ** frac_bits`` as the fractional scale;
* fusion weights are applied in the fixed-point integer domain in the fixed
  detector order (maha, residual, crest) and rounded half-up;
* the fused total is rescaled back to the configured integer score interval,
  clipped, and compared with the threshold (equality uses ``tie_rule`` and is
  flagged as a boundary position).
"""

from dataclasses import dataclass

import numpy as np

from .detectors import DETECTOR_ORDER
from .hashing import combine_keys, stable_hash


def quant_param_hash(cfg):
    return stable_hash([
        "quant",
        cfg["normalization"],
        cfg["quantization"],
        cfg["fusion"],
        cfg["score"],
    ])


def normalize(scores, cfg):
    method = cfg["normalization"]["method"]
    epsilon = float(cfg["normalization"]["epsilon"])
    values = np.asarray(scores, dtype=np.float64)
    if method == "minmax":
        lo, hi = float(np.min(values)), float(np.max(values))
        if hi - lo <= epsilon:
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)
    if method == "robust":
        median = float(np.median(values))
        scale = float(np.median(np.abs(values - median)) * 1.4826)
        if scale <= epsilon:
            return np.zeros_like(values)
        return (values - median) / scale
    if method == "maxabs":
        hi = float(np.max(np.abs(values)))
        if hi <= epsilon:
            return np.zeros_like(values)
        return values / hi
    raise ValueError(f"unknown normalization method: {method}")


def to_fixed(normalized, frac_bits, mode="round"):
    scale = float(2 ** frac_bits)
    scaled = np.clip(np.asarray(normalized, dtype=np.float64), 0.0, 1.0) * scale
    if mode == "truncate":
        return np.floor(scaled).astype(np.int64)
    return np.floor(scaled + 0.5).astype(np.int64)


def _fused_total(fixed_by_name, weights, frac_bits, score_cfg):
    position_count = fixed_by_name[DETECTOR_ORDER[0]].shape[0]
    totals = np.zeros(position_count, dtype=np.int64)
    for index, name in enumerate(DETECTOR_ORDER):
        totals += np.floor(
            weights[index] * fixed_by_name[name].astype(np.float64) + 0.5
        ).astype(np.int64)
    totals = np.floor(totals / float(2 ** frac_bits) *
                      float(score_cfg["max"]) + 0.5).astype(np.int64)
    return np.clip(totals, int(score_cfg["min"]), int(score_cfg["max"]))


@dataclass
class PositionVerdict:
    position: int
    normalized: dict
    fixed: dict
    total: int
    threshold: int
    is_anomaly: bool
    boundary: bool
    reference_total: int
    reference_frac_bits: int
    reference_delta: int
    discrepancy_cause: str
    verdict_key: str

    def to_dict(self):
        return {
            "position": self.position,
            "normalized": {k: float(v) for k, v in self.normalized.items()},
            "fixed": dict(self.fixed),
            "total": int(self.total),
            "threshold": int(self.threshold),
            "is_anomaly": bool(self.is_anomaly),
            "boundary": bool(self.boundary),
            "reference_total": int(self.reference_total),
            "reference_frac_bits": int(self.reference_frac_bits),
            "reference_delta": int(self.reference_delta),
            "discrepancy_cause": self.discrepancy_cause,
            "verdict_key": self.verdict_key,
        }


def _decide(value, threshold, tie_rule):
    if value > threshold:
        return True
    if value < threshold:
        return False
    return tie_rule == "anomaly"


def fuse_and_judge(detector_results, cfg):
    frac_bits = int(cfg["quantization"]["frac_bits"])
    ref_bits = frac_bits + int(cfg["quantization"]["ref_extra_bit"])
    weights = [float(w) for w in cfg["fusion"]["weights"]]
    if len(weights) != len(DETECTOR_ORDER):
        raise ValueError("fusion weights must match the three detectors")
    threshold = int(cfg["score"]["threshold"])
    tie_rule = cfg["score"]["tie_rule"]
    q_hash = quant_param_hash(cfg)

    fixed_by_name, norm_by_name, ref_by_name, trunc_ref_by_name = {}, {}, {}, {}
    for name in DETECTOR_ORDER:
        normalized = normalize(detector_results[name].scores, cfg)
        fixed_by_name[name] = to_fixed(normalized, frac_bits, "round")
        ref_by_name[name] = to_fixed(normalized, ref_bits, "round")
        trunc_ref_by_name[name] = to_fixed(normalized, ref_bits, "truncate")
        norm_by_name[name] = normalized

    totals = _fused_total(fixed_by_name, weights, frac_bits, cfg["score"])
    ref_totals = _fused_total(ref_by_name, weights, ref_bits, cfg["score"])
    trunc_totals = _fused_total(trunc_ref_by_name, weights, ref_bits,
                                cfg["score"])

    verdicts = []
    for p in range(totals.shape[0]):
        total = int(totals[p])
        ref_total = int(ref_totals[p])
        trunc_total = int(trunc_totals[p])
        is_anomaly = _decide(total, threshold, tie_rule)
        boundary = total == threshold
        ref_decision = _decide(ref_total, threshold, tie_rule)
        trunc_decision = _decide(trunc_total, threshold, tie_rule)
        cause = "none"
        if boundary and ref_decision != is_anomaly:
            if trunc_decision != is_anomaly:
                cause = "quantization_truncation"
            else:
                cause = "rounding"
        verdict_key = combine_keys(
            stable_hash(["verdict", p]),
            stable_hash([detector_results[name].cache_keys[p]
                         for name in DETECTOR_ORDER]),
            q_hash)
        verdicts.append(PositionVerdict(
            position=p,
            normalized={name: float(norm_by_name[name][p])
                        for name in DETECTOR_ORDER},
            fixed={name: int(fixed_by_name[name][p]) for name in DETECTOR_ORDER},
            total=total, threshold=threshold, is_anomaly=is_anomaly,
            boundary=boundary, reference_total=ref_total,
            reference_frac_bits=ref_bits,
            reference_delta=ref_total - total,
            discrepancy_cause=cause, verdict_key=verdict_key))
    return verdicts
