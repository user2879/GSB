"""Local NumPy lightweight forward pass with integer fixed-point outputs."""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import numpy as np

# Fixed-point scale for per-request outputs (micro units, 1e6).
OUTPUT_SCALE = 1_000_000


def forward_scores(params: Tuple[float, ...],
                   inputs: Tuple[Tuple[float, ...], ...],
                   repeat: int = 1) -> np.ndarray:
    """score_i = dot(input_i, params); repeat>1 only burns deterministic CPU."""
    matrix = np.asarray(inputs, dtype=np.float64)
    vector = np.asarray(params, dtype=np.float64)
    scores = matrix @ vector
    for _ in range(max(0, repeat - 1)):
        scores = matrix @ vector
    return scores


def summarize(scores: np.ndarray) -> Dict[int, int]:
    """Per-request integer fixed-point summary (round-half-even via rint)."""
    return {i: int(np.rint(score * OUTPUT_SCALE))
            for i, score in enumerate(scores)}


def batch_digest(summaries: Dict[int, int]) -> str:
    """Deterministic digest of a batch's integer summaries, index ordered."""
    payload = ",".join(str(summaries[i]) for i in sorted(summaries))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
