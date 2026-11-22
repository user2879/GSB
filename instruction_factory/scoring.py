from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .constants import DIMENSION_ORDER
from .tokenizer import SubwordTokenizer
from .util import as_decimal, canonical_json, decimal_text, fingerprint_object, stable_key_hash


@dataclass(frozen=True)
class QuantizedScore:
    fixed: int
    display: str
    residual: Decimal


@dataclass(frozen=True)
class Quantizer:
    places: int
    step: Decimal

    @property
    def quantum_units(self) -> int:
        return int(self.step * (Decimal(10) ** self.places))

    @property
    def fingerprint(self) -> str:
        return fingerprint_object({"places": self.places, "step": str(self.step)})

    def quantize(self, raw: Decimal) -> QuantizedScore:
        units = (raw / self.step).quantize(Decimal(1), rounding=ROUND_HALF_UP)
        fixed = int(units) * self.quantum_units
        scale = Decimal(fixed).scaleb(-self.places)
        if self.places:
            display = f"{scale:.{self.places}f}"
        else:
            display = str(fixed)
        return QuantizedScore(fixed=fixed, display=display, residual=raw - scale)


def quantizer_from_config(config: dict[str, Any]) -> Quantizer:
    return Quantizer(
        places=int(config["fixed_decimal_places"]),
        step=as_decimal(config["quantization_step"]),
    )


def _clamp(value: Decimal, low: Decimal = Decimal(0), high: Decimal = Decimal(100)) -> Decimal:
    return max(low, min(high, value))


def score_cost(sampling: dict[str, Any]) -> int:
    source_payload = canonical_json(sampling["slot_values"])
    length = len(sampling["instruction"]) + len(sampling["reference_answer"]) + len(source_payload)
    return 1 + length // 24


def score_parameter_fingerprint(
    scoring: dict[str, Any],
    quantizer: Quantizer,
    tokenizer: SubwordTokenizer,
) -> str:
    return fingerprint_object(
        {
            "scoring": scoring,
            "quantizer": {"places": quantizer.places, "step": str(quantizer.step)},
            "tokenizer": tokenizer.rules,
        }
    )


def scoring_cache_key(
    *,
    sampling: dict[str, Any],
    scoring: dict[str, Any],
    quantizer: Quantizer,
    tokenizer: SubwordTokenizer,
) -> tuple[str, str, int]:
    parameter_fingerprint = score_parameter_fingerprint(scoring, quantizer, tokenizer)
    threshold = quantizer.quantize(as_decimal(scoring["pass_threshold"]))
    cache_key = stable_key_hash(
        "scoring-v1",
        sampling["candidate_id"],
        sampling["content_fingerprint"],
        parameter_fingerprint,
        str(threshold.fixed),
    )
    return cache_key, parameter_fingerprint, threshold.fixed


def score_candidate(
    *,
    sampling: dict[str, Any],
    scoring: dict[str, Any],
    quantizer: Quantizer,
    tokenizer: SubwordTokenizer,
) -> dict[str, Any]:
    instruction = sampling["instruction"]
    reference = sampling["reference_answer"]
    instruction_tokens = set(tokenizer.tokenize(instruction))
    reference_tokens = set(tokenizer.tokenize(reference))
    token_count = len(tokenizer.tokenize(instruction))
    unique_count = len(instruction_tokens)
    duplicate_rate = (
        (Decimal(token_count - unique_count) / Decimal(token_count))
        if token_count
        else Decimal(0)
    )
    terminal = instruction.rstrip().endswith(("。", "！", "？", ".", "!", "?"))
    braces_clear = "{" not in instruction and "}" not in instruction
    slots_rendered = all(value in instruction for value in sampling["slot_values"].values())
    overlap = (
        Decimal(len(instruction_tokens & reference_tokens)) / Decimal(len(reference_tokens))
        if reference_tokens
        else Decimal(0)
    )
    ideal_length = int(scoring["ideal_length_chars"])
    penalty_scale = as_decimal(scoring["length_penalty_scale"])

    raw_dimensions: dict[str, Decimal] = {}
    for dimension in DIMENSION_ORDER:
        baseline = as_decimal(scoring["dimensions"][dimension]["baseline"])
        if dimension == "clarity":
            value = baseline
            value += Decimal(5) if terminal else Decimal(0)
            value += Decimal(3) if 4 <= token_count <= 40 else Decimal(0)
            value -= Decimal(8) * duplicate_rate
        elif dimension == "correctness":
            value = baseline
            value += Decimal(12) if reference.strip() else Decimal(0)
            value += Decimal(8) * overlap
        elif dimension == "format":
            value = baseline
            value += Decimal(5) if braces_clear else Decimal(0)
            value += Decimal(5) if terminal else Decimal(0)
            value += Decimal(5) if slots_rendered else Decimal(0)
        else:
            deviation = abs(token_count - ideal_length)
            value = baseline + Decimal(6) - min(Decimal(30), Decimal(deviation) * penalty_scale)
            value += Decimal(2) if 8 <= len(instruction) <= 200 else Decimal(0)
        raw_dimensions[dimension] = _clamp(value)

    weighted = Decimal(0)
    weight_total = Decimal(0)
    for dimension in DIMENSION_ORDER:
        weight = as_decimal(scoring["dimensions"][dimension]["weight"])
        weighted += weight * raw_dimensions[dimension]
        weight_total += weight
    raw_total = weighted / weight_total

    dimension_scores: dict[str, Any] = {}
    for dimension in DIMENSION_ORDER:
        quantized = quantizer.quantize(raw_dimensions[dimension])
        dimension_scores[dimension] = {
            "raw": decimal_text(raw_dimensions[dimension]),
            "fixed": quantized.fixed,
            "display": quantized.display,
            "residual": decimal_text(quantized.residual),
        }
    total = quantizer.quantize(raw_total)
    threshold = quantizer.quantize(as_decimal(scoring["pass_threshold"]))
    boundary = total.fixed == threshold.fixed
    quality_pass = total.fixed >= threshold.fixed
    cache_key, parameter_fingerprint, _ = scoring_cache_key(
        sampling=sampling,
        scoring=scoring,
        quantizer=quantizer,
        tokenizer=tokenizer,
    )
    return {
        "stage": "scoring",
        "candidate_id": sampling["candidate_id"],
        "task_type": sampling["task_type"],
        "sample_seq": sampling["sample_seq"],
        "stable_sort_key": f"{sampling['candidate_id']}|scoring",
        "sampling_key": sampling["cache_key"],
        "parameter_fingerprint": parameter_fingerprint,
        "dimensions": dimension_scores,
        "raw_total": decimal_text(raw_total),
        "total_fixed": total.fixed,
        "total_display": total.display,
        "quantization_residual": decimal_text(total.residual),
        "threshold_fixed": threshold.fixed,
        "threshold_display": threshold.display,
        "quality_pass": quality_pass,
        "boundary": boundary,
        "boundary_decision": "keep_and_order_by_candidate_id" if boundary else None,
        "score_cost": score_cost(sampling),
        "cache_key": cache_key,
    }
