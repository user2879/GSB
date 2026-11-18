"""Stage-wise error propagation between configured width and the higher reference width."""
from __future__ import annotations

from typing import Any

import numpy as np

from .aggregation import Aggregation
from .embedding import EmbeddingResult
from .fixedpoint import FixedPoint
from .similarity import PairSimilarity
from .util import diff_metrics

STAGE_ORDER = ("feature_quantization", "statistic_aggregation",
               "embedding_projection", "similarity")
TOLERANCE_KEY = {
    "feature_quantization": "quantization",
    "statistic_aggregation": "aggregation",
    "embedding_projection": "embedding",
    "similarity": "similarity",
}


def feature_quantization_diff(base_emb: EmbeddingResult, ref_emb: EmbeddingResult) -> dict[str, Any]:
    """Difference of the dequantized per-feature quantization outputs at the two widths."""
    base = FixedPoint(base_emb.width).dequantize(base_emb.feature_q_ints)
    ref = FixedPoint(ref_emb.width).dequantize(ref_emb.feature_q_ints)
    return diff_metrics(base, ref)


def aggregation_diff(base_agg: Aggregation, ref_agg: Aggregation) -> dict[str, Any]:
    """Upstream float64 statistics are width independent; the diff must be exactly zero."""
    return diff_metrics(base_agg.stats_vector, ref_agg.stats_vector)


def embedding_diff(base_emb: EmbeddingResult, ref_emb: EmbeddingResult) -> dict[str, Any]:
    return diff_metrics(base_emb.embedding_float, ref_emb.embedding_float)


def similarity_diff(base_pair: PairSimilarity | None, ref_pair: PairSimilarity | None) -> dict[str, Any]:
    if base_pair is None or ref_pair is None:
        return {
            "element_count": 1,
            "per_element_abs_diff": [],
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "rms_diff": 0.0,
            "available": False,
        }
    return diff_metrics(
        np.array([base_pair.similarity], dtype=np.float64),
        np.array([ref_pair.similarity], dtype=np.float64),
    )


def build_sample_error_report(
    sample_id: str,
    base_agg: Aggregation,
    ref_agg: Aggregation,
    base_emb: EmbeddingResult,
    ref_emb: EmbeddingResult,
    tolerance: dict[str, float],
) -> dict[str, Any]:
    stages = {
        "feature_quantization": feature_quantization_diff(base_emb, ref_emb),
        "statistic_aggregation": aggregation_diff(base_agg, ref_agg),
        "embedding_projection": embedding_diff(base_emb, ref_emb),
    }
    exceeded = {
        name: stages[name]["max_abs_diff"] > tolerance[TOLERANCE_KEY[name]]
        for name in stages
    }
    first_stage = next((name for name in STAGE_ORDER[:3] if exceeded[name]), None)
    return {
        "sample_id": sample_id,
        "width_base": base_emb.width,
        "width_reference": ref_emb.width,
        "stages": stages,
        "tolerance": dict(tolerance),
        "exceeded": exceeded,
        "first_exceeded_stage": first_stage,
    }


def build_pair_error_report(
    pair_id: str,
    sample_report_a: dict[str, Any],
    sample_report_b: dict[str, Any],
    base_pair: PairSimilarity | None,
    ref_pair: PairSimilarity | None,
    tolerance: dict[str, float],
) -> dict[str, Any]:
    sim = similarity_diff(base_pair, ref_pair)
    sim_exceeded = sim["max_abs_diff"] > tolerance[TOLERANCE_KEY["similarity"]]

    # Contradiction rule: embedding within tolerance but downstream similarity exceeds it,
    # or vice versa. The downstream (similarity) verdict always wins.
    emb_exceeded = bool(
        sample_report_a["exceeded"]["embedding_projection"]
        or sample_report_b["exceeded"]["embedding_projection"]
    )
    adjudication = None
    if emb_exceeded != sim_exceeded:
        adjudication = {
            "embedding_exceeded": emb_exceeded,
            "similarity_exceeded": sim_exceeded,
            "winning_stage": "similarity",
            "reason": (
                "embedding-level and similarity-level tolerance verdicts contradict; "
                "the more downstream similarity stage is authoritative because the "
                "integer dot-product/norm reduction can amplify or absorb component "
                "differences non-uniformly"
            ),
            "final_exceeded": bool(sim_exceeded),
        }

    pair_first = None
    for name in STAGE_ORDER:
        if name == "similarity":
            if sim_exceeded:
                pair_first = name
        elif name == "embedding_projection":
            if emb_exceeded and pair_first is None:
                pair_first = name
        elif (sample_report_a["exceeded"].get(name, False)
              or sample_report_b["exceeded"].get(name, False)) and pair_first is None:
            pair_first = name

    return {
        "pair_id": pair_id,
        "stages": {
            "feature_quantization": _combine(sample_report_a, sample_report_b,
                                             "feature_quantization"),
            "statistic_aggregation": _combine(sample_report_a, sample_report_b,
                                              "statistic_aggregation"),
            "embedding_projection": _combine(sample_report_a, sample_report_b,
                                             "embedding_projection"),
            "similarity": sim,
        },
        "exceeded": {
            "feature_quantization": bool(
                sample_report_a["exceeded"]["feature_quantization"]
                or sample_report_b["exceeded"]["feature_quantization"]
            ),
            "statistic_aggregation": bool(
                sample_report_a["exceeded"]["statistic_aggregation"]
                or sample_report_b["exceeded"]["statistic_aggregation"]
            ),
            "embedding_projection": emb_exceeded,
            "similarity": bool(sim_exceeded),
        },
        "first_exceeded_stage": pair_first,
        "adjudication": adjudication,
        "tolerance": dict(tolerance),
    }


def _combine(report_a: dict[str, Any], report_b: dict[str, Any], stage: str) -> dict[str, Any]:
    a = report_a["stages"][stage]
    b = report_b["stages"][stage]
    return {
        "sample_a_max_abs_diff": a["max_abs_diff"],
        "sample_b_max_abs_diff": b["max_abs_diff"],
        "max_abs_diff": max(a["max_abs_diff"], b["max_abs_diff"]),
        "mean_abs_diff": float((a["mean_abs_diff"] + b["mean_abs_diff"]) / 2.0),
        "element_count": a["element_count"] + b["element_count"],
    }
