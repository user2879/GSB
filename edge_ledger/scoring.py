from __future__ import annotations

import math
from dataclasses import dataclass

from .config import PipelineConfig
from .errors import Failure
from .loader import ValidatedSample
from .models import QualityBreakdown


SCORE_SCALE = 1_000_000


class ScoreFailure(RuntimeError):
    def __init__(self, message: str, *, injection_id: str | None = None) -> None:
        super().__init__(message)
        self.code = "E_SCORE_NONFINITE"
        self.stage = "score"
        self.injection_id = injection_id

    def as_failure(self, sample_id: str) -> Failure:
        return Failure(
            code=self.code,
            stage=self.stage,
            message=str(self),
            target_kind="sample",
            target_id=sample_id,
            injection_id=self.injection_id,
        )


def _round_half_up_integer(value: int, step: int) -> int:
    if step <= 0:
        raise ValueError("quantum step must be positive")
    return min(SCORE_SCALE, ((value + step // 2) // step) * step)


def compute_quality(
    sample: ValidatedSample,
    answer_text: str,
    config: PipelineConfig,
) -> QualityBreakdown:
    slots = sample.instruction.slots
    total_slots = len(slots)
    if total_slots == 0:
        completeness = SCORE_SCALE
    else:
        values = sample.slot_values()
        filled = sum(1 for slot in slots if values.get(slot.rule.name, "") != "")
        completeness = (filled * SCORE_SCALE) // total_slots

    used_placeholders = set(sample.instruction.instruction_placeholders) | set(
        sample.answer.answer_placeholders
    )
    if used_placeholders:
        values = sample.slot_values()
        covered = sum(1 for name in used_placeholders if name in values)
        template_coverage = (covered * SCORE_SCALE) // len(used_placeholders)
    else:
        template_coverage = SCORE_SCALE

    excess = max(0, len(answer_text) - config.answer_soft_limit_chars)
    answer_length = max(
        config.answer_floor,
        SCORE_SCALE - excess * config.answer_decay_per_char,
    )
    answer_length = min(SCORE_SCALE, answer_length)

    components: dict[str, int] = {
        "completeness": completeness,
        "template_coverage": template_coverage,
        "answer_length": answer_length,
    }
    weights = config.weights_permille
    raw_score = (
        completeness * weights["completeness"]
        + template_coverage * weights["template_coverage"]
        + answer_length * weights["answer_length"]
    ) // 1000

    if sample.sample.quality_override == "NaN":
        # This deliberately enters the real non-finite validation path rather than
        # producing a synthetic validation failure at load time.
        injected = raw_score * float("nan")
        if not math.isfinite(injected):
            raise ScoreFailure(
                "quality computation produced a non-finite value",
                injection_id=sample.sample.injection_ids[0] if sample.sample.injection_ids else None,
            )
        raw_score = int(injected)

    raw_score = max(0, min(SCORE_SCALE, raw_score))
    quantized = _round_half_up_integer(raw_score, config.quantum_step)
    return QualityBreakdown(
        raw_score=raw_score,
        quantized_score=quantized,
        quantum_step=config.quantum_step,
        residual=raw_score - quantized,
        components=components,
        decimal_places=config.decimal_places,
        quantization=config.quantization,
    )

