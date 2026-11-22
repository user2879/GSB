"""Fixed-point multimodal fusion with explicit evidence relation semantics."""

from __future__ import annotations

from .config import PipelineConfig
from .fixedpoint import (
    scale,
    from_float,
    fmul,
    fdiv_round,
    softmax_two,
    clamp_confidence,
    quant_shift,
)
from .hashing import cache_key, combine
from .models import (
    RegionFeature,
    RegionResult,
    Contribution,
    RELATION_CONFLICT,
    RELATION_SUPPLEMENT,
    RELATION_LINK,
    CATEGORY_SUPPORT,
    CATEGORY_REJECT,
    CATEGORY_NEUTRAL,
)

INTRINSIC = "intrinsic"


def fusion_compute_key(
    feature: RegionFeature, config: PipelineConfig, relation_hash: str
) -> str:
    """Key for the threshold-independent fixed-point fusion computation."""
    return cache_key(
        "fusion-compute",
        combine(
            feature.cache_key,
            relation_hash,
            config.fingerprint(),
            config.quant_level,
            config.frac_bits,
            config.fusion_temperature,
        ),
    )


def fusion_cache_key(
    feature: RegionFeature, config: PipelineConfig, relation_hash: str
) -> str:
    """Full result key, including the decision-only confidence threshold.

    Threshold changes therefore invalidate only the final verdict artifact;
    feature and fusion-compute artifacts stay reusable.
    """
    return cache_key(
        "fusion-result",
        combine(
            fusion_compute_key(feature, config, relation_hash),
            config.confidence_threshold,
        ),
    )


def _text_scalar(
    feature: RegionFeature, sample, config: PipelineConfig
) -> tuple[int, list[Contribution]]:
    s = scale(config.frac_bits)
    contributions: list[Contribution] = []
    if not feature.text_vector:
        return 0, contributions
    magnitude = sum(abs(x) for x in feature.text_vector)
    # Normalize per dimension and re-scale once: keeps the scalar inside the
    # unit fixed-point range instead of accumulating dim-wide magnitudes.
    effective = fdiv_round(magnitude, config.text_dim * scale(config.frac_bits),
                           config.frac_bits)

    relation_types = sorted(
        {
            r.relation_type
            for r in sample.relations
            if r.region == feature.region
        }
    )
    if not relation_types:
        contributions.append(
            Contribution(
                modality="text",
                relation_type=INTRINSIC,
                assertion_id=None,
                weight_fixed=s,
                score_fixed=effective,
            )
        )
        return effective, contributions

    if RELATION_CONFLICT in relation_types:
        discount = from_float(config.conflict_discount, config.frac_bits)
        effective = fmul(effective, discount, config.frac_bits)
        contributions.append(
            Contribution(
                modality="text",
                relation_type=RELATION_CONFLICT,
                assertion_id=None,
                weight_fixed=discount,
                score_fixed=effective,
            )
        )
    if RELATION_SUPPLEMENT in relation_types:
        boost = from_float(config.supplement_boost, config.frac_bits)
        effective = fmul(effective, boost, config.frac_bits)
        contributions.append(
            Contribution(
                modality="text",
                relation_type=RELATION_SUPPLEMENT,
                assertion_id=None,
                weight_fixed=boost,
                score_fixed=effective,
            )
        )
    if RELATION_LINK in relation_types:
        contributions.append(
            Contribution(
                modality="text",
                relation_type=RELATION_LINK,
                assertion_id=None,
                weight_fixed=s,
                score_fixed=effective,
            )
        )
    return effective, contributions


def _visual_scalar(
    feature: RegionFeature, config: PipelineConfig
) -> tuple[int, Contribution]:
    magnitude = sum(abs(x) for x in feature.visual_vector)
    score = fdiv_round(magnitude, config.visual_dim * scale(config.frac_bits),
                       config.frac_bits)
    weight = from_float(config.visual_weight, config.frac_bits)
    weighted = fmul(score, weight, config.frac_bits)
    return weighted, Contribution(
        modality="visual",
        relation_type=INTRINSIC,
        assertion_id=None,
        weight_fixed=weight,
        score_fixed=weighted,
    )


def _quantized_equal(
    visual_weighted: int, text_weighted: int, config: PipelineConfig
) -> bool:
    """Modality equality decided after applying the level's quantization shift.

    Using the shift (and not a display-layer rounding) is what makes boundary
    detection genuinely dependent on the quantization level enum.
    """
    shift = quant_shift(config.quant_level)
    return (visual_weighted >> shift) == (text_weighted >> shift)


def fuse_region(
    feature: RegionFeature, sample, config: PipelineConfig
) -> RegionResult:
    s = scale(config.frac_bits)

    visual_weighted, visual_contribution = _visual_scalar(feature, config)
    text_base, text_contribs = _text_scalar(feature, sample, config)
    text_weight = from_float(config.text_weight, config.frac_bits)
    text_weighted = fmul(text_base, text_weight, config.frac_bits)
    # Weighted-text headline contribution; detailed conflict/supplement/link
    # entries already follow in a fixed relation-type ordering.
    text_contribs.insert(
        0,
        Contribution(
            modality="text",
            relation_type=RELATION_LINK if text_contribs else INTRINSIC,
            assertion_id=None,
            weight_fixed=text_weight,
            score_fixed=text_weighted,
        ),
    )

    temperature = from_float(config.fusion_temperature, config.frac_bits)
    w_visual, w_text = softmax_two(
        visual_weighted, text_weighted, config.frac_bits, temperature
    )
    # w_visual/w_text are fixed-point weights summing to S; this is an exact
    # integer weighted average with no double scaling.
    blended = (visual_weighted * w_visual + text_weighted * w_text) // s
    confidence = clamp_confidence(blended, config.frac_bits)

    threshold_fixed = from_float(config.confidence_threshold, config.frac_bits)
    boundary_reasons: list[str] = []
    shift = quant_shift(config.quant_level)
    visual_quantized = visual_weighted >> shift
    text_quantized = text_weighted >> shift
    threshold_tie = confidence == threshold_fixed
    modality_tie = visual_quantized == text_quantized
    if threshold_tie:
        boundary_reasons.append("threshold_tie")
    if modality_tie:
        boundary_reasons.append("modality_tie")
    boundary = bool(boundary_reasons)

    if threshold_tie:
        # Fixed deterministic decision for an exact threshold hit.
        category = CATEGORY_REJECT
    elif modality_tie:
        # Equal modality contributions (post-quantization) -> neutral.
        category = CATEGORY_NEUTRAL
    elif confidence > threshold_fixed:
        category = CATEGORY_SUPPORT
    else:
        category = CATEGORY_REJECT

    contributions = [visual_contribution] + text_contribs
    key = fusion_cache_key(feature, config, sample.relation_hash)
    return RegionResult(
        sample_id=feature.sample_id,
        region=feature.region,
        category=category,
        confidence_fixed=confidence,
        visual_score_fixed=visual_weighted,
        text_score_fixed=text_weighted,
        visual_weight_fixed=w_visual,
        text_weight_fixed=w_text,
        visual_quantized=visual_quantized,
        text_quantized=text_quantized,
        contributions=contributions,
        boundary=boundary,
        boundary_reasons=boundary_reasons,
        referenced_assertions=list(feature.referenced_assertions),
        referenced_regions=list(feature.referenced_regions),
        feature_cache_key=feature.cache_key,
        cache_key=key,
    )
