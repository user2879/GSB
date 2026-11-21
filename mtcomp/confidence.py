from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import (
    fixed_from_float,
    fixed_sigmoid_margin,
    fixed_to_permille,
)
from .decoder import DecodeResult

ACCEPT = "ACCEPT"
REVIEW = "REVIEW"
REJECT = "REJECT"


@dataclass(frozen=True)
class ConfidenceResult:
    confidence_key: str
    confidence_fixed: int
    confidence_permille: int
    threshold_fixed: int
    threshold_permille: int
    status: str
    boundary: bool
    fixed_point_bits: int
    scale: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfidenceResult":
        return cls(
            confidence_key=data["confidence_key"],
            confidence_fixed=int(data["confidence_fixed"]),
            confidence_permille=int(data["confidence_permille"]),
            threshold_fixed=int(data["threshold_fixed"]),
            threshold_permille=int(data["threshold_permille"]),
            status=data["status"],
            boundary=bool(data["boundary"]),
            fixed_point_bits=int(data["confidence_fixed_bits"]),
            scale=int(data["confidence_fixed_scale"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence_key": self.confidence_key,
            "confidence_fixed": self.confidence_fixed,
            "confidence_permille": self.confidence_permille,
            "confidence_fixed_bits": self.fixed_point_bits,
            "confidence_fixed_scale": self.scale,
            "threshold_fixed": self.threshold_fixed,
            "threshold_permille": self.threshold_permille,
            "status": self.status,
            "boundary": self.boundary,
            "equality_rule": "confidence == threshold => REVIEW",
        }


def classify_confidence(confidence_fixed: int, threshold_fixed: int) -> str:
    if confidence_fixed > threshold_fixed:
        return ACCEPT
    if confidence_fixed < threshold_fixed:
        return REJECT
    return REVIEW


def score_confidence(
    decode: DecodeResult,
    *,
    confidence_key: str,
    threshold: float,
    bits: int,
    tier: int,
) -> ConfidenceResult:
    scale = 1 << bits
    margin_ratio = fixed_sigmoid_margin(decode.margin_fixed, bits)
    raw = (3 * decode.mean_alignment_conf_fixed + margin_ratio) // 4
    step = max(1, scale // tier)
    confidence = max(0, min(scale, ((raw + step // 2) // step) * step))
    threshold_fixed = max(0, min(scale, fixed_from_float(threshold, bits, tier)))
    status = classify_confidence(confidence, threshold_fixed)
    return ConfidenceResult(
        confidence_key=confidence_key,
        confidence_fixed=confidence,
        confidence_permille=fixed_to_permille(confidence, bits),
        threshold_fixed=threshold_fixed,
        threshold_permille=fixed_to_permille(threshold_fixed, bits),
        status=status,
        boundary=confidence == threshold_fixed,
        fixed_point_bits=bits,
        scale=scale,
    )


def boundary_detail(
    base: ConfidenceResult,
    reference: ConfidenceResult,
    *,
    base_path: tuple[tuple[Any, ...], ...],
    reference_path: tuple[tuple[Any, ...], ...],
) -> dict[str, Any]:
    flip = base.status != reference.status
    if not flip:
        cause = "NONE"
    elif base_path != reference_path:
        cause = "ALIGNMENT_QUANTIZATION_RESIDUAL"
    else:
        cause = "DECODE_ROUNDING_ACCUMULATION"
    return {
        "reference_fixed_bits": reference.fixed_point_bits,
        "reference_confidence_fixed": reference.confidence_fixed,
        "reference_confidence_permille": reference.confidence_permille,
        "reference_confidence_fixed_scale": reference.scale,
        "reference_status": reference.status,
        "delta_fixed": reference.confidence_fixed - base.confidence_fixed,
        "delta_permille": reference.confidence_permille - base.confidence_permille,
        "decision_flip": flip,
        "flip_cause": cause,
    }
