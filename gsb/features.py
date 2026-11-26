from __future__ import annotations

from typing import Any


def subword_set(sample: dict[str, Any]) -> frozenset[str]:
    """Extract deterministic subword features.

    The generator's token keys are normalized word pieces/subwords.  Only
    positive-count pieces participate so zero counts cannot perturb similarity.
    """

    counts = sample["token_counts"]
    features = {str(term).lower() for term, count in counts.items() if count > 0}
    explicit = sample.get("subword_features")
    if explicit is not None:
        features.update(str(item) for item in explicit)
    return frozenset(features)


def jaccard_numerator_denominator(left: frozenset[str], right: frozenset[str]) -> tuple[int, int]:
    intersection = len(left & right)
    union = len(left | right)
    return intersection, union


def similarity_permille(intersection: int, union: int) -> int:
    if union == 0:
        return 1000
    return (intersection * 1000) // union
