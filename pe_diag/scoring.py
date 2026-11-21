"""Dot-product attention scoring and in/out-of-training layering.

For one (scheme, sample, precision) combination the module:

1. builds the scheme encoder with the strategy-selected active dimension;
2. rotates the query/key feature vectors;
3. computes scaled dot-product scores with one of two genuine compute paths:
   * ``single`` builds every key score in one vector then softmaxes;
   * ``stream`` walks keys one-by-one with an online (running max/sum) softmax;
4. labels every key by relative distance: ``inside`` when
   ``|key-query| <= train_len`` (equality always resolves to the inside region
   by the fixed boundary rule) and ``extrap`` otherwise;
5. returns one uniform :class:`ScoreResult` for downstream aggregation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .common import resolve_dtype
from .encoding import PositionEncoder
from .errors import E_KEY_POS_OOR, E_NONFINITE_INPUT, DiagError


@dataclass
class ScoreResult:
    scheme: str
    sample_id: str
    tier_id: str
    precision: str
    query_pos: int
    key_positions: tuple[int, ...]
    rel_distances: tuple[int, ...]
    regions: tuple[str, ...]
    raw_scores: np.ndarray          # (K,)
    scaled_scores: np.ndarray       # (K,)
    weights: np.ndarray             # (K,)
    active_dim: int
    block_method: str
    layer_precision: str

    def to_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "sample_id": self.sample_id,
            "tier_id": self.tier_id,
            "precision": self.precision,
            "query_pos": self.query_pos,
            "key_positions": list(self.key_positions),
            "rel_distances": list(self.rel_distances),
            "regions": list(self.regions),
            "raw_scores": self.raw_scores,
            "scaled_scores": self.scaled_scores,
            "weights": self.weights,
            "active_dim": self.active_dim,
            "block_method": self.block_method,
            "layer_precision": self.layer_precision,
        }


def _check_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        raise DiagError(E_NONFINITE_INPUT,
                        "non-finite entry in {}".format(name))


def softmax_single(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def softmax_stream(scores: np.ndarray) -> np.ndarray:
    """Online softmax (Milakov & Gimelshein).

    The running max and denominator are corrected each step and the stored
    exponentials are rescaled by the accumulated max correction at the end.
    Its sequential accumulation rounds differently from the batched path in
    float32 while converging to the same float64 result."""
    running_max = scores[0].copy()
    running_sum = np.zeros((), dtype=scores.dtype)
    for value in scores:
        new_max = np.maximum(running_max, value)
        running_sum = (
            running_sum * np.exp(
                np.array(running_max - new_max, dtype=scores.dtype))
            + np.exp(np.array(value - new_max, dtype=scores.dtype)))
        running_max = new_max
    # second sequential pass: each exp and each divide accumulate in order
    out = np.empty_like(scores)
    for idx, value in enumerate(scores):
        out[idx] = np.exp(np.array(value - running_max,
                                   dtype=scores.dtype)) / running_sum
    return out


def label_regions(query_pos: int, key_positions: tuple[int, ...],
                  train_len: int, target_len: int) -> tuple[tuple[int, ...],
                                                            tuple[str, ...]]:
    rels: list[int] = []
    regions: list[str] = []
    for key_pos in key_positions:
        if key_pos < 0 or key_pos > target_len:
            raise DiagError(
                E_KEY_POS_OOR,
                "key position {} outside [0, target_len={}]".format(
                    key_pos, target_len),
            )
        rel = abs(int(key_pos) - int(query_pos))
        rels.append(rel)
        # fixed boundary rule: equality belongs to the inside region
        regions.append("inside" if rel <= train_len else "extrap")
    return tuple(rels), tuple(regions)


def score_sample(sample, config, decision: dict, precision: str,
                 *, base_override: float | None = None,
                 scale_override: float | None = None,
                 feature_overrides: dict | None = None) -> ScoreResult:
    """Compute one :class:`ScoreResult`.

    ``feature_overrides`` lets the fault-injection layer feed genuinely
    corrupted inputs (non-finite vectors); base/scale overrides drive both
    injection and the sensitivity scans.
    """
    scheme = decision["scheme"]
    active_dim = int(decision["active_dim"])
    block_method = decision["block_method"]
    layer_precision = decision["layer_precision"]
    base = config.base if base_override is None else float(base_override)
    scale = config.scale if scale_override is None else float(scale_override)
    dtype = resolve_dtype(precision)

    # faults applied to the real feature input before any math
    q_feat = np.asarray(sample.query_features, dtype=np.float64).copy()
    k_feat = np.asarray(sample.key_features, dtype=np.float64).copy()
    if feature_overrides:
        for target in feature_overrides.get("query", []):
            q_feat[target["component"]] = target["value"]
        for entry in feature_overrides.get("key", []):
            k_feat[entry["key_index"], entry["component"]] = entry["value"]
    _check_finite("query_features", q_feat)
    _check_finite("key_features", k_feat)

    rels, regions = label_regions(
        sample.query_pos, sample.key_positions,
        config.train_len, config.target_len)

    encoder = PositionEncoder(
        scheme=scheme,
        base=base,
        train_len=config.train_len,
        target_len=config.target_len,
        dim=config.dim,
        scale=scale,
        chunk_size=config.chunk_size,
        precision=precision,
        active_dim=active_dim,
    )

    all_positions = [sample.query_pos, *sample.key_positions]
    encoder.encode_positions(all_positions)  # validates overflow/non-finite

    q_rot = encoder.rotate(q_feat.astype(dtype), sample.query_pos)
    q_use = q_rot[:active_dim]
    k_rot = np.stack([
        encoder.rotate(k_feat[i].astype(dtype), sample.key_positions[i])
        for i in range(len(sample.key_positions))
    ])
    k_use = k_rot[:, :active_dim]

    scale_denom = math.sqrt(active_dim)
    if block_method == "stream":
        raw_scores = np.empty(k_use.shape[0], dtype=dtype)
        for i in range(k_use.shape[0]):
            raw_scores[i] = float(np.dot(q_use, k_use[i]))
        scaled = (raw_scores / scale_denom).astype(dtype)
        weights = softmax_stream(scaled)
    else:
        raw_scores = k_use @ q_use
        scaled = (raw_scores / scale_denom).astype(dtype)
        weights = softmax_single(scaled)

    if not (np.all(np.isfinite(raw_scores)) and np.all(np.isfinite(weights))):
        raise DiagError(E_NONFINITE_INPUT,
                        "non-finite score/weight for sample {}".format(
                            sample.sample_id))

    return ScoreResult(
        scheme=scheme,
        sample_id=sample.sample_id,
        tier_id=sample.tier_id,
        precision=precision,
        query_pos=sample.query_pos,
        key_positions=tuple(sample.key_positions),
        rel_distances=rels,
        regions=regions,
        raw_scores=raw_scores.astype(np.float64),
        scaled_scores=scaled.astype(np.float64),
        weights=weights.astype(np.float64),
        active_dim=active_dim,
        block_method=block_method,
        layer_precision=layer_precision,
    )
