"""Intent flip attribution across revisions, gears and dedup thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .dedup import DedupMember
from .scoring import ScoreResult

# Attribution source codes, ordered by diagnostic precedence.
FLIP_DEDUP = "DEDUP_CLUSTER_MERGE"
FLIP_THRESHOLD = "THRESHOLD_COMPARISON"
FLIP_QUANT = "FIXED_POINT_ROUNDING"
FLIP_WEIGHT = "TRIGGER_WEIGHT_CHANGE"
FLIP_NONE = "NO_FLIP"


@dataclass
class FlipEvidence:
    text_id: str
    before_intent: str
    after_intent: str
    flipped: bool
    source: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _top_path(score: ScoreResult) -> tuple[str, int, int]:
    """Return (intent, reference_raw, quantized_raw) for the decided path."""
    if not score.paths:
        return score.decided_intent, 0, 0
    intent = score.decided_intent
    path = score.paths.get(intent)
    if path is None:
        return intent, 0, 0
    return intent, path.reference_raw, path.quantized_raw


def _path_map(score: ScoreResult) -> dict[str, tuple[int, int]]:
    return {
        intent: (path.reference_raw, path.quantized_raw)
        for intent, path in score.paths.items()
    }


def attribute_flip(
    text_id: str,
    before: ScoreResult,
    after: ScoreResult,
    before_dedup: DedupMember | None,
    after_dedup: DedupMember | None,
) -> FlipEvidence:
    """Attribute an authoritative fixed-point decision change to one source.

    Precedence (first matching wins, each with numeric evidence):
      1. dedup cluster membership changed (representative suppression),
      2. threshold eligibility changed while reference scores are equal,
      3. quantized decision differs while reference scores stay equal,
      4. otherwise the trigger library / weights changed the reference score.
    The final conclusion always uses the quantized fixed-point integer only.
    """
    before_intent = before.decided_intent
    after_intent = after.decided_intent
    _, before_ref, before_quant = _top_path(before)
    _, after_ref, after_quant = _top_path(after)
    before_map = _path_map(before)
    after_map = _path_map(after)
    # Reference scores are library-driven; compare intent-by-intent.
    reference_unchanged = (
        set(before_map) == set(after_map)
        and all(
            before_map[intent][0] == after_map[intent][0]
            for intent in before_map
        )
    )

    base_evidence = {
        "before_reference_raw": before_ref,
        "after_reference_raw": after_ref,
        "reference_delta": after_ref - before_ref,
        "before_quantized_raw": before_quant,
        "after_quantized_raw": after_quant,
        "quantized_delta": after_quant - before_quant,
        "quantization_error_before": before_ref - before_quant,
        "quantization_error_after": after_ref - after_quant,
    }

    if before_intent == after_intent:
        return FlipEvidence(
            text_id=text_id,
            before_intent=before_intent,
            after_intent=after_intent,
            flipped=False,
            source=FLIP_NONE,
            evidence=base_evidence,
        )

    # 1. Dedup-driven: the text changed representative/suppression status.
    before_rep = before_dedup.representative_id if before_dedup else None
    after_rep = after_dedup.representative_id if after_dedup else None
    if before_rep != after_rep:
        return FlipEvidence(
            text_id, before_intent, after_intent, True, FLIP_DEDUP,
            {
                **base_evidence,
                "before_representative": before_rep,
                "after_representative": after_rep,
                "before_similarity": (
                    before_dedup.similarity_permille if before_dedup else None
                ),
                "after_similarity": (
                    after_dedup.similarity_permille if after_dedup else None
                ),
            },
        )

    # 2. Threshold eligibility change with identical reference scoring.
    before_elig = set(before.eligible)
    after_elig = set(after.eligible)
    if reference_unchanged and before_elig != after_elig:
        return FlipEvidence(
            text_id, before_intent, after_intent, True, FLIP_THRESHOLD,
            {
                **base_evidence,
                "before_eligible": sorted(before_elig),
                "after_eligible": sorted(after_elig),
            },
        )

    # 3. Quantization-driven: identical reference math, quantized output moved.
    quant_changed = any(
        before_map.get(intent, (0, 0))[0]
        == after_map.get(intent, (0, 0))[0]
        and before_map.get(intent, (0, 0))[1]
        != after_map.get(intent, (0, 0))[1]
        for intent in set(before_map) | set(after_map)
    )
    if reference_unchanged and (quant_changed or before_quant != after_quant):
        return FlipEvidence(
            text_id, before_intent, after_intent, True, FLIP_QUANT,
            base_evidence,
        )

    # 4. Trigger/weight change moved the high-bit-width reference score.
    return FlipEvidence(
        text_id, before_intent, after_intent, True, FLIP_WEIGHT,
        {
            **base_evidence,
            "changed_paths": sorted(
                set(before.paths) ^ set(after.paths)
            ),
        },
    )
