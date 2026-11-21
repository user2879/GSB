"""Stage-wise extrapolation error tracking.

The reference is the in-training (unscaled) rotary encoding computed in
float64.  For every extrapolation key we measure three successive deviations:

1. position-vector relative L2 norm against the reference rotary vector;
2. dot-product score deviation against the reference score;
3. normalized attention-weight deviation against the reference softmax.

The first stage whose deviation exceeds the configured tolerance is reported as
``first_breach_stage``.  All deviations are computed in float64 with full
precision.  When the float32 and float64 final weights disagree beyond the
weight tolerance, the cause is split between *position-encoding dimension
truncation* and *score-accumulation rounding* via two explicit, reportable
separation measures.
"""

from __future__ import annotations

import math

import numpy as np

from .common import resolve_dtype
from .encoding import PositionEncoder
from .scoring import label_regions, score_sample, softmax_single

CAUSE_TRUNCATION = "position_encoding_dimension_truncation"
CAUSE_ROUNDING = "score_accumulation_rounding"


def _rel_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = actual.astype(np.float64)
    reference = reference.astype(np.float64)
    denom = float(np.linalg.norm(reference))
    if denom == 0.0:
        return float(np.linalg.norm(actual - reference))
    return float(np.linalg.norm(actual - reference) / denom)


def _reference_scoring(sample, config, decision) -> dict:
    """Unscaled full-dimension float64 RoPE reference for the same sample."""
    active_dim = int(decision["active_dim"])
    dtype = resolve_dtype("float64")
    q_feat = np.asarray(sample.query_features, dtype=np.float64)
    k_feat = np.asarray(sample.key_features, dtype=np.float64)
    ref_encoder = PositionEncoder(
        scheme="rope", base=config.base, train_len=config.train_len,
        target_len=config.target_len, dim=config.dim, scale=1.0,
        chunk_size=config.dim, precision="float64", active_dim=active_dim,
    )
    q_rot = ref_encoder.rotate(q_feat.astype(dtype), sample.query_pos)[:active_dim]
    k_rot = np.stack([
        ref_encoder.rotate(k_feat[i].astype(dtype), sample.key_positions[i])
        for i in range(len(sample.key_positions))
    ])[:, :active_dim]
    raw = k_rot @ q_rot
    scaled = raw / math.sqrt(active_dim)
    weights = softmax_single(scaled.astype(np.float64))
    enc = ref_encoder.encode_positions(list(sample.key_positions))
    return {
        "raw": raw.astype(np.float64),
        "scaled": scaled.astype(np.float64),
        "weights": weights.astype(np.float64),
        "vectors": enc.position_vectors.astype(np.float64),
        "encoder": ref_encoder,
    }


def track_sample(sample, config, decision, result64: dict,
                 result32: dict | None) -> dict:
    scheme = decision["scheme"]
    active_dim = int(decision["active_dim"])
    tolerances = config.tolerances
    ref = _reference_scoring(sample, config, decision)

    # actual scheme encoder (float64) for position-vector comparison
    actual_encoder = PositionEncoder(
        scheme=scheme, base=config.base, train_len=config.train_len,
        target_len=config.target_len, dim=config.dim, scale=config.scale,
        chunk_size=config.chunk_size, precision="float64",
        active_dim=active_dim,
    )
    actual_enc = actual_encoder.encode_positions(list(sample.key_positions))

    rels, regions = label_regions(
        sample.query_pos, sample.key_positions,
        config.train_len, config.target_len)

    records = []
    first_breach_overall = None
    for i, region in enumerate(regions):
        if region != "extrap":
            continue
        pos_dev = _rel_l2(actual_enc.position_vectors[i],
                          ref["vectors"][i])
        score_dev = abs(float(result64["scaled_scores"][i]) -
                        float(ref["scaled"][i]))
        weight_dev = abs(float(result64["weights"][i]) -
                         float(ref["weights"][i]))
        breaches = []
        if pos_dev > tolerances["position"]:
            breaches.append("position_vector")
        if score_dev > tolerances["score"]:
            breaches.append("score")
        if weight_dev > tolerances["weight"]:
            breaches.append("weight")
        first_breach = breaches[0] if breaches else "within_tolerance"
        if first_breach_overall is None and breaches:
            first_breach_overall = first_breach
        records.append({
            "key_index": i,
            "key_position": sample.key_positions[i],
            "rel_distance": rels[i],
            "region": region,
            "position_vector_deviation": pos_dev,
            "score_deviation": score_dev,
            "weight_deviation": weight_dev,
            "breached_stages": breaches,
            "first_breach_stage": first_breach,
        })

    precision = _precision_attribution(
        sample, config, decision, result64, result32, tolerances)

    return {
        "scheme": scheme,
        "sample_id": sample.sample_id,
        "tier_id": sample.tier_id,
        "active_dim": active_dim,
        "reference": {"scheme": "rope", "precision": "float64",
                      "scale": 1.0, "dim_range": [0, active_dim]},
        "tolerances": dict(tolerances),
        "first_breach_stage": (first_breach_overall or "within_tolerance"),
        "extrap_records": records,
        "precision_inconsistency": precision,
    }


def _precision_attribution(sample, config, decision, result64, result32,
                           tolerances) -> dict:
    if result32 is None:
        return {"inconsistent": False, "cause": None, "measures": None}
    w64 = np.asarray(result64["weights"], dtype=np.float64)
    w32 = np.asarray(result32["weights"], dtype=np.float64)
    max_weight_gap = float(np.max(np.abs(w64 - w32)))
    inconsistent = max_weight_gap > tolerances["weight"]
    if not inconsistent:
        return {"inconsistent": False, "cause": None,
                "measures": {"max_weight_gap": max_weight_gap,
                             "truncation_gap": 0.0,
                             "rounding_gap": 0.0}}

    active_dim = int(decision["active_dim"])
    # truncation gap: full-dimension float64 vs active-dimension float64,
    # same single-block accumulation -> isolates dimension truncation.
    full_decision = dict(decision)
    full_decision["active_dim"] = config.dim
    full_decision["block_method"] = "single"
    full_result = score_sample(sample, config, full_decision, "float64")
    truncation_gap = float(np.max(np.abs(
        np.asarray(full_result.weights) - w64)))

    # rounding gap: active-dimension float64 single vs float32 stream, holding
    # the active dimension fixed -> isolates accumulation rounding.
    round_decision = dict(decision)
    round_decision["block_method"] = "stream"
    r64_stream = score_sample(sample, config, round_decision, "float64")
    r32_stream = score_sample(sample, config, round_decision, "float32")
    rounding_gap = float(np.max(np.abs(
        np.asarray(r64_stream.weights) - np.asarray(r32_stream.weights))))

    if truncation_gap >= rounding_gap:
        cause = CAUSE_TRUNCATION
    else:
        cause = CAUSE_ROUNDING
    return {
        "inconsistent": True,
        "cause": cause,
        "measures": {
            "max_weight_gap": max_weight_gap,
            "truncation_gap": truncation_gap,
            "rounding_gap": rounding_gap,
        },
    }
