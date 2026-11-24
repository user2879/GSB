"""Three deterministic detectors sharing one feature-spectrum input.

Detectors are pure functions of the spectrum matrix plus their own
configuration sections; no runtime randomness is involved.  Per-position
cache keys include both the position key and a fingerprint of the device-level
spectrum statistics (median/robust scale/baseline) because those statistics
are estimated across the whole matrix and change when any position changes.
"""

from dataclasses import dataclass

import numpy as np

from .hashing import array_fingerprint, combine_keys, stable_hash

DETECTOR_ORDER = ("maha", "residual", "crest")


@dataclass
class DetectorResult:
    name: str
    scores: np.ndarray
    param_hash: str
    cache_keys: list

    def to_manifest(self):
        return {
            "name": self.name,
            "param_hash": self.param_hash,
            "score_fingerprint": array_fingerprint(self.scores),
            "keys": list(self.cache_keys),
        }


def detector_param_hash(cfg, name):
    return stable_hash([f"detector:{name}", cfg["detectors"][name]])


def _spectrum_context_hash(matrix):
    median = np.median(matrix, axis=0)
    robust = np.median(np.abs(matrix - median), axis=0) * 1.4826
    return stable_hash([
        "spectrum-context",
        array_fingerprint(median),
        array_fingerprint(robust),
        int(matrix.shape[0]),
    ])


def _per_position_key(result_prefix, position_cache_key, param_hash,
                      context_hash, p):
    return combine_keys(
        stable_hash([result_prefix, p]), position_cache_key, param_hash,
        context_hash)


def maha_scores(matrix, cfg, position_keys, context_hash):
    params = cfg["detectors"]["maha"]
    epsilon = float(params["scale_epsilon"])
    multiplier = float(params["scale_multiplier"])
    median = np.median(matrix, axis=0)
    deviation = np.abs(matrix - median)
    scale = np.median(deviation, axis=0) * 1.4826
    scale = np.where(scale > epsilon, scale, epsilon)
    standardized = (matrix - median) / scale
    squared = standardized * standardized
    distances = np.sqrt(np.add.reduce(squared, axis=1)) * multiplier
    param_hash = detector_param_hash(cfg, "maha")
    keys = [_per_position_key("maha", position_keys[p], param_hash,
                              context_hash, p)
            for p in range(matrix.shape[0])]
    return DetectorResult("maha", distances.astype(np.float64), param_hash, keys)


def residual_scores(matrix, cfg, position_keys, context_hash):
    params = cfg["detectors"]["residual"]
    floor = float(params["floor"])
    power = float(params["power"])
    baseline = np.median(matrix, axis=0)
    baseline = np.where(baseline > floor, baseline, floor)
    ratio = matrix / baseline
    mean_ratio = np.add.reduce(ratio, axis=1) / matrix.shape[1]
    scores = (np.abs(mean_ratio - 1.0) + floor) ** power
    param_hash = detector_param_hash(cfg, "residual")
    keys = [_per_position_key("residual", position_keys[p], param_hash,
                              context_hash, p)
            for p in range(matrix.shape[0])]
    return DetectorResult("residual", scores.astype(np.float64), param_hash, keys)


def crest_scores(matrix, cfg, position_keys, context_hash):
    params = cfg["detectors"]["crest"]
    epsilon = float(params["epsilon"])
    extreme = np.max(np.abs(matrix), axis=0)
    dispersion = np.median(np.abs(matrix - np.median(matrix, axis=0)),
                           axis=0) * 1.4826
    dispersion = np.where(dispersion > epsilon, dispersion, epsilon)
    crest = extreme / dispersion
    per_position = np.add.reduce(crest * np.abs(matrix), axis=1) / (
        np.add.reduce(np.abs(matrix), axis=1) + epsilon)
    scores = per_position + epsilon
    param_hash = detector_param_hash(cfg, "crest")
    keys = [_per_position_key("crest", position_keys[p], param_hash,
                              context_hash, p)
            for p in range(matrix.shape[0])]
    return DetectorResult("crest", scores.astype(np.float64), param_hash, keys)


def run_detectors(matrix, positions, cfg):
    position_keys = [pos.cache_key for pos in positions]
    context_hash = _spectrum_context_hash(matrix)
    return {
        "maha": maha_scores(matrix, cfg, position_keys, context_hash),
        "residual": residual_scores(matrix, cfg, position_keys, context_hash),
        "crest": crest_scores(matrix, cfg, position_keys, context_hash),
    }
