"""Stable fixed-point weighted fusion and threshold decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import FusionConfig, QuantizationProfile, make_profile
from .util import clamp, exp_negative_q, hash_json, q_to_string, round_div, saturate_signed


@dataclass(frozen=True)
class FusionStage:
    score_q: int
    threshold_q: int
    decision: str
    boundary: bool
    contributions: list[dict[str, Any]]
    modality_contributions: list[dict[str, Any]]
    accumulation_order: list[str]
    max_mean: int
    profile: QuantizationProfile

    def as_dict(self) -> dict[str, Any]:
        return {
            "score_q": self.score_q,
            "score": q_to_string(self.score_q, self.profile.fractional_bits),
            "score_range": [0, self.profile.one],
            "threshold_q": self.threshold_q,
            "threshold": q_to_string(self.threshold_q, self.profile.fractional_bits),
            "decision": self.decision,
            "boundary_sample": self.boundary,
            "decision_rule": self.decision_rule,
            "contributions": self.contributions,
            "modality_contributions": self.modality_contributions,
            "accumulation_order": self.accumulation_order,
            "max_mean": self.max_mean,
            "profile": self.profile.as_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FusionStage":
        profile_dict = value["profile"]
        profile = make_profile(
            int(profile_dict["tier"]),
            int(profile_dict["fractional_bits"]),
            int(profile_dict["word_bits"]),
            int(profile_dict["quant_step"]) << int(profile_dict["tier"]),
        )
        return cls(
            score_q=int(value["score_q"]),
            threshold_q=int(value["threshold_q"]),
            decision=str(value["decision"]),
            boundary=bool(value["boundary_sample"]),
            contributions=value["contributions"],
            modality_contributions=value["modality_contributions"],
            accumulation_order=value["accumulation_order"],
            max_mean=int(value["max_mean"]),
            profile=profile,
        )

    @property
    def decision_rule(self) -> str:
        if self.threshold_q == self.score_q and self.decision == "negative":
            return "score > threshold => positive; equality resolves negative"
        return "score >= threshold => positive"


def fusion_stage_key(
    visual: Any,
    text: Any,
    config: FusionConfig,
    profile: QuantizationProfile,
) -> str:
    return hash_json(
        {
            "schema": "fusion-stage-v1",
            "fusion_config_hash": config.fusion_config_hash,
            "visual_cache_key": visual.visual_cache_key,
            "text_cache_key": text.text_cache_key,
            "profile": profile.as_dict(),
        },
        namespace="fusion-stage-key-v1",
    )


def fuse_and_decide(
    visual_vector: list[int],
    text_vector: list[int],
    config: FusionConfig,
    profile: QuantizationProfile,
) -> FusionStage:
    one = profile.one
    vectors = {"text": text_vector, "visual": visual_vector}
    means: dict[str, int] = {}
    for modality in sorted(vectors):
        total = 0
        for dimension in range(len(vectors[modality])):
            total += int(vectors[modality][dimension])
        means[modality] = clamp(
            round_div(total, len(vectors[modality])) if vectors[modality] else 0,
            -one,
            one,
        )

    max_mean = max(means[modality] for modality in sorted(means))
    temperature_q = config.temperature_q(profile)
    if temperature_q <= 0:
        temperature_q = 1
    masses: dict[str, int] = {}
    exponentials: dict[str, int] = {}
    for modality in sorted(means):
        scaled = round_div((means[modality] - max_mean) * one, temperature_q)
        exponential = exp_negative_q(scaled, one)
        exponentials[modality] = exponential
        weight_q = config.weight_q(modality, profile)
        masses[modality] = round_div(exponential * weight_q, one)
    mass_total = sum(masses[modality] for modality in sorted(masses))
    if mass_total <= 0:
        for modality in sorted(means):
            masses[modality] = config.weight_q(modality, profile)
        mass_total = sum(masses[modality] for modality in sorted(masses))

    modality_contributions: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    accumulation_order: list[str] = []
    score = 0
    for modality in sorted(vectors):
        probability = round_div(masses[modality] * one, mass_total) if mass_total else 0
        activation = clamp(round_div(means[modality] + one, 2), 0, one)
        score_contribution = round_div(probability * activation, one)
        score += score_contribution
        modality_contributions.append(
            {
                "modality": modality,
                "weight_q": config.weight_q(modality, profile),
                "mean_q": means[modality],
                "max_subtracted_delta_q": means[modality] - max_mean,
                "exp_q": exponentials[modality],
                "softmax_probability_q": probability,
                "activation_q": activation,
                "score_contribution_q": score_contribution,
            }
        )
        for dimension in range(len(vectors[modality])):
            accumulation_order.append(f"{modality}:{dimension}")
            value = int(vectors[modality][dimension])
            contributions.append(
                {
                    "modality": modality,
                    "dimension": dimension,
                    "value_q": value,
                    "activation_q": clamp(round_div(value + one, 2), 0, one),
                }
            )

    score = clamp(score, 0, one)
    score, _ = saturate_signed(score, profile.word_bits)
    score = clamp(score, 0, one)
    threshold_q = clamp(config.threshold_q(profile), 0, one)
    boundary = score == threshold_q
    if config.tie_break == "negative":
        positive = score > threshold_q
    else:
        positive = score >= threshold_q
    return FusionStage(
        score_q=score,
        threshold_q=threshold_q,
        decision="positive" if positive else "negative",
        boundary=boundary,
        contributions=contributions,
        modality_contributions=modality_contributions,
        accumulation_order=accumulation_order,
        max_mean=max_mean,
        profile=profile,
    )
