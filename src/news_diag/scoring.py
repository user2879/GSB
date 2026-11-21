from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, localcontext

import numpy as np

from .models import Document, Sentence, SentenceScore, TermScore, WorkbenchConfig


def _clamp_decimal(value: Decimal, lower: Decimal, upper: Decimal) -> Decimal:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def _quantize_weighted(
    name: str,
    normalized: Decimal,
    weight: Decimal,
    config: WorkbenchConfig,
    reference_value: float,
) -> TermScore:
    weighted = normalized * weight
    level_decimal = Decimal(config.quantization_levels)
    units = (weighted * level_decimal).to_integral_value(rounding=ROUND_HALF_UP)
    fixed_integer = int(units) * config.quantization_unit
    fixed_value = Decimal(fixed_integer) / Decimal(config.score_scale)
    residual = fixed_value - weighted
    return TermScore(
        name=name,
        normalized=normalized,
        weight=weight,
        weighted=weighted,
        fixed_integer=fixed_integer,
        fixed_value=fixed_value,
        rounding_residual=residual,
        reference=reference_value,
    )


def score_sentence(
    document: Document,
    sentence: Sentence,
    sentence_count: int,
    frequency_total: Decimal,
    config: WorkbenchConfig,
    *,
    frequency_delta: Decimal = Decimal("0"),
    entity_coverage_delta: Decimal = Decimal("0"),
) -> SentenceScore:
    with localcontext() as context:
        context.prec = 50
        effective_frequency = max(Decimal("0"), sentence.frequency_hits + frequency_delta)
        if frequency_total > 0:
            frequency_normalized = _clamp_decimal(
                effective_frequency / frequency_total, Decimal("0"), Decimal("1")
            )
        else:
            frequency_normalized = Decimal("0")

        entity_total = len(document.entities)
        entity_hits = len(sentence.entity_ids)
        base_entity_ratio = Decimal(entity_hits) / Decimal(entity_total)
        entity_normalized = _clamp_decimal(
            base_entity_ratio + entity_coverage_delta, Decimal("0"), Decimal("1")
        )

        if sentence_count <= 1:
            position_normalized = Decimal("1")
        else:
            position_normalized = Decimal("1") - (
                Decimal(sentence.sentence_index) / Decimal(sentence_count - 1)
            )
        position_normalized = _clamp_decimal(
            position_normalized, Decimal("0"), Decimal("1")
        )

        # This float64 sequence deliberately mirrors the fixed-point term order.
        frequency_ref = (
            np.float64(float(effective_frequency) / float(frequency_total))
            if frequency_total > 0
            else np.float64(0.0)
        )
        frequency_ref = np.float64(min(1.0, max(0.0, float(frequency_ref))))
        entity_ref = np.float64(
            min(1.0, max(0.0, float(entity_hits) / float(entity_total)))
        )
        entity_ref = np.float64(
            min(1.0, max(0.0, float(entity_ref) + float(entity_coverage_delta)))
        )
        position_ref = np.float64(
            max(
                0.0,
                min(
                    1.0,
                    1.0
                    - (
                        float(sentence.sentence_index) / float(sentence_count - 1)
                        if sentence_count > 1
                        else 0.0
                    ),
                ),
            )
        )

        frequency_weighted_ref = np.float64(
            frequency_ref * np.float64(float(config.word_frequency_weight))
        )
        entity_weighted_ref = np.float64(
            entity_ref * np.float64(float(config.entity_coverage_weight))
        )
        position_weighted_ref = np.float64(
            position_ref * np.float64(float(config.position_decay_coefficient))
        )

        frequency_term = _quantize_weighted(
            "word_frequency",
            frequency_normalized,
            config.word_frequency_weight,
            config,
            float(frequency_weighted_ref),
        )
        entity_term = _quantize_weighted(
            "entity_coverage",
            entity_normalized,
            config.entity_coverage_weight,
            config,
            float(entity_weighted_ref),
        )
        position_term = _quantize_weighted(
            "position_decay",
            position_normalized,
            config.position_decay_coefficient,
            config,
            float(position_weighted_ref),
        )
        terms = (frequency_term, entity_term, position_term)
        fixed_integer = sum(term.fixed_integer for term in terms)
        fixed_value = Decimal(fixed_integer) / Decimal(config.score_scale)
        ideal_total = sum((term.weighted for term in terms), Decimal("0"))
        residual = fixed_value - ideal_total
        reference_total = np.float64(
            np.float64(frequency_weighted_ref + entity_weighted_ref)
            + position_weighted_ref
        )
        return SentenceScore(
            fixed_integer=fixed_integer,
            fixed_value=fixed_value,
            ideal_total=ideal_total,
            reference=float(reference_total),
            residual=residual,
            terms=terms,
            frequency_hits=effective_frequency,
            frequency_total=frequency_total,
            entity_hits=entity_hits,
            entity_total=entity_total,
            entity_delta=entity_coverage_delta,
        )
