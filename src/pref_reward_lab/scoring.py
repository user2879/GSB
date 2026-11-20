from __future__ import annotations

import math
from typing import Mapping

from .canonical import fingerprint, stable_float
from .config import LabConfig
from .errors import LabError
from .models import FeatureResult, ScoreResult


def _segment_signal(counts: Mapping[int, int]) -> float:
    terms: list[float] = []
    for bucket in sorted(counts):
        polarity = ((bucket * 2_654_435_761) % 2001 - 1000) / 1000.0
        terms.append(float(counts[bucket]) * polarity)
    return math.fsum(terms)


def _score_answer(
    *,
    prompt_signal: float,
    answer_signal: float,
    answer_tokens: int,
    weights: Mapping[str, float],
    answer_kind: str,
) -> float:
    terms = [
        float(weights["prompt"]) * prompt_signal,
        float(weights[answer_kind]) * answer_signal,
        float(weights["length"]) * math.log1p(answer_tokens),
        float(weights["bias"]),
    ]
    return stable_float(math.fsum(terms))


def score_features(
    features: FeatureResult,
    config: LabConfig,
    *,
    effective_threshold: float,
    weights_override: Mapping[str, float] | None = None,
    boundary_equal_threshold: bool = False,
    injection_id: str | None = None,
) -> ScoreResult:
    weights = dict(config.scoring_weights)
    if weights_override:
        weights.update({key: float(value) for key, value in weights_override.items()})
    nonfinite = sorted(key for key, value in weights.items() if not math.isfinite(float(value)))
    if nonfinite:
        raise LabError(
            "E_NONFINITE_WEIGHT",
            f"non-finite scoring weight(s): {', '.join(nonfinite)}",
            {
                "pair_id": features.pair_id,
                "stage": "scoring",
                "injection_id": injection_id,
                "fields": nonfinite,
            },
        )
    prompt_signal = _segment_signal(features.prompt_buckets)
    chosen_signal = _segment_signal(features.chosen_buckets)
    rejected_signal = _segment_signal(features.rejected_buckets)
    chosen_score = _score_answer(
        prompt_signal=prompt_signal,
        answer_signal=chosen_signal,
        answer_tokens=features.chosen_tokens,
        weights=weights,
        answer_kind="chosen",
    )
    rejected_score = _score_answer(
        prompt_signal=prompt_signal,
        answer_signal=rejected_signal,
        answer_tokens=features.rejected_tokens,
        weights=weights,
        answer_kind="rejected",
    )
    reward_delta = stable_float(chosen_score - rejected_score)
    if boundary_equal_threshold:
        reward_delta = stable_float(effective_threshold)
    boundary = reward_delta <= effective_threshold
    return ScoreResult(
        pair_id=features.pair_id,
        chosen_score=chosen_score,
        rejected_score=rejected_score,
        reward_delta=reward_delta,
        effective_threshold=stable_float(effective_threshold),
        boundary=boundary,
    )


def scoring_cache_key(
    features: FeatureResult,
    config: LabConfig,
    *,
    effective_hash_buckets: int,
    precision_tier: str,
    effective_threshold: float,
    source_version: str,
    injection_fingerprint: str = "",
) -> str:
    return fingerprint(
        {
            "features": features.content_dict(),
            "scoring_weights": config.scoring_weights,
            "effective_threshold": effective_threshold,
            "effective_hash_buckets": effective_hash_buckets,
            "precision_tier": precision_tier,
            "source_version": source_version,
            "injection": injection_fingerprint,
            "scorer_schema": 1,
        },
        prefix="scoring:",
    )

