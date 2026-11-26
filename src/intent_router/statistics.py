"""Intent distribution statistics with deterministic numpy aggregation.

numpy is used purely for integer tallying; inputs and outputs are plain Python
integers so cross-platform determinism is preserved.
"""

from __future__ import annotations

import numpy as np


def fixed_point_distribution(
    ordered_intents: list[str], decisions: list[str]
) -> dict[str, int]:
    """Count decisions per intent using a fixed integer index space."""
    index = {intent: i for i, intent in enumerate(ordered_intents)}
    counts = np.zeros(len(ordered_intents), dtype=np.int64)
    for decision in decisions:
        if decision in index:
            counts[index[decision]] += 1
    return {
        intent: int(counts[index[intent]]) for intent in ordered_intents
    }


def permille_share(distribution: dict[str, int]) -> dict[str, int]:
    """Integer parts-per-thousand share of each intent (floor, remainder-safe)."""
    total = sum(distribution.values())
    if total == 0:
        return {intent: 0 for intent in distribution}
    raw = np.array(
        [distribution[intent] * 1000 for intent in distribution], dtype=np.int64
    )
    values = (raw // total).tolist()
    return dict(zip(distribution.keys(), (int(v) for v in values)))
