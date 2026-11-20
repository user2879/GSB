"""Three-stage extrapolation error tracing, all in float64.

Reference: the scheme's own position encoding evaluated at the position
wrapped into the training range (``pos % train_len``) -- i.e. the encoding
the scheme produces for positions inside the training context length.

Stages, checked in order:
1. ``position_vector`` -- relative L2 deviation of the key embedding.
2. ``score``           -- relative deviation of the dot-product score.
3. ``weight``          -- L1 deviation of max-subtracted softmax weights.

The first stage whose deviation exceeds the configured tolerance is
recorded as ``first_exceeded_stage`` (null when all stages pass).
"""
from __future__ import annotations

import numpy as np

from .schemes import SchemeParams, position_embedding, rotate_features

STAGES = ("position_vector", "score", "weight")


def _softmax_maxsub(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def trace_sample(
    params: SchemeParams,
    query: np.ndarray,
    keys: np.ndarray,
    query_pos: int,
    key_positions: list[int],
    rotated_query: np.ndarray,
    rotated_keys: np.ndarray,
    scores: np.ndarray,
    tolerance: float,
    dim_range: tuple[int, int] | None = None,
) -> dict:
    """Trace deviations for the extrapolation keys of one sample."""
    query = np.asarray(query, dtype=np.float64)
    keys = np.asarray(keys, dtype=np.float64)
    train_len = params.train_len

    extra_idx = [
        i for i, kp in enumerate(key_positions) if abs(query_pos - int(kp)) > train_len
    ]
    records = []
    if extra_idx:
        ref_positions = np.array(
            [int(key_positions[i]) % train_len for i in extra_idx], dtype=np.float64
        )
        ref_key_emb = position_embedding(ref_positions, params)
        ref_rot_keys = rotate_features(keys[extra_idx], ref_positions, params)

        q = query
        if dim_range is not None:
            lo, hi = dim_range
            q = q[lo:hi]
            ref_rot_keys = ref_rot_keys[:, lo:hi]
        ref_scores = ref_rot_keys @ q

        act_key_emb = position_embedding(
            np.array([key_positions[i] for i in extra_idx], dtype=np.float64), params
        )
        act_scores = np.asarray(scores, dtype=np.float64)[extra_idx]

        ref_weights = _softmax_maxsub(ref_scores)
        act_weights = _softmax_maxsub(act_scores)

        for row, i in enumerate(extra_idx):
            ref_vec = ref_key_emb[row]
            act_vec = act_key_emb[row]
            vec_dev = float(
                np.linalg.norm(act_vec - ref_vec) / (np.linalg.norm(ref_vec) + 1e-300)
            )
            ref_score = float(ref_scores[row])
            act_score = float(act_scores[row])
            score_dev = abs(act_score - ref_score) / (abs(ref_score) + 1e-300)
            weight_dev = float(abs(act_weights[row] - ref_weights[row]))
            deviations = {
                "position_vector": vec_dev,
                "score": float(score_dev),
                "weight": weight_dev,
            }
            first_exceeded = next(
                (s for s in STAGES if deviations[s] > tolerance), None
            )
            records.append(
                {
                    "key_position": int(key_positions[i]),
                    "rel_distance": abs(int(query_pos) - int(key_positions[i])),
                    "deviations": deviations,
                    "first_exceeded_stage": first_exceeded,
                }
            )
    exceeded = [r["first_exceeded_stage"] for r in records if r["first_exceeded_stage"]]
    return {
        "tolerance": float(tolerance),
        "n_extrapolation_keys": len(records),
        "records": records,
        "first_exceeded_stage": exceeded[0] if exceeded else None,
        "stage_exceed_counts": {s: exceeded.count(s) for s in STAGES},
    }
