"""Preprocessing: missing-value imputation, category encoding, scaling.

All statistics are accumulated in a fixed row order with math.fsum so two
runs over the same input produce element-identical outputs. The artifact
carries a content fingerprint derived from the column-description
fingerprint and the parameter fingerprint.
"""

from __future__ import annotations

import bisect
import math
from typing import Any

import numpy as np

from .config_loader import DataDescription, ExperimentConfig
from .fingerprint import fingerprint

QUANTILE_TIE_RULE = "average_rank_linear_clip"


def _fsum_mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _fsum_std(values: list[float], mean: float) -> float:
    if not values:
        return 0.0
    var = math.fsum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _quantile_transform(sorted_train: list[float], values: list[float]) -> list[float]:
    """Map values to [0, 1] by empirical CDF with a fixed tie rule.

    Tie rule (QUANTILE_TIE_RULE): tied values share the average of their
    sorted positions; interpolation between reference points is linear;
    out-of-range values are clipped to [0, 1].
    """
    n = len(sorted_train)
    if n == 0:
        return [0.0 for _ in values]
    if n == 1:
        return [0.0 if v <= sorted_train[0] else 1.0 for v in values]
    out = []
    for v in values:
        left = bisect.bisect_left(sorted_train, v)
        right = bisect.bisect_right(sorted_train, v)
        if left == right:  # not exactly present: linear interpolation
            if left == 0:
                q = 0.0
            elif left >= n:
                q = 1.0
            else:
                lo, hi = sorted_train[left - 1], sorted_train[left]
                frac = 0.0 if hi == lo else (v - lo) / (hi - lo)
                q = ((left - 1) + frac) / (n - 1)
        else:  # average rank of the tied block
            pos = (left + right - 1) / 2.0
            q = pos / (n - 1)
        out.append(min(1.0, max(0.0, q)))
    return out


def _numeric_like(col) -> bool:
    for v in col:
        if v is None:
            continue
        return isinstance(v, (int, float, np.integer, np.floating))
    return True


def preprocess(desc: DataDescription, cfg: ExperimentConfig,
               data: dict[str, Any], data_fp: str) -> dict[str, Any]:
    """Fit on the train portion (first rows, fixed order) and transform all."""
    n_rows = data["n_rows"]
    n_train = max(1, int(n_rows * desc.train_fraction))
    train_idx = list(range(n_train))  # fixed accumulation order

    feature_names: list[str] = []
    stats: dict[str, Any] = {}
    columns_out: list[list[float]] = []

    for spec in desc.columns:  # fixed column order
        col = data["columns"][spec.name]
        if spec.type == "numeric" or (
                spec.type == "redundant" and _numeric_like(col)):
            raw = [None if v is None else float(v) for v in col]
            train_vals = [raw[i] for i in train_idx if raw[i] is not None]
            mean = _fsum_mean(train_vals)
            std = _fsum_std(train_vals, mean)
            fill = mean
            filled = [fill if v is None else v for v in raw]
            n_missing = sum(1 for v in raw if v is None)
            if cfg.preprocess_method == "standard":
                scale = std if std > 0.0 else 1.0
                out = [(v - mean) / scale for v in filled]
                stats[spec.name] = {"kind": "numeric", "method": "standard",
                                    "mean": mean, "std": std, "fill": fill,
                                    "n_missing": n_missing}
            else:  # quantile
                sorted_train = sorted(filled[i] for i in train_idx)
                out = _quantile_transform(sorted_train, filled)
                stats[spec.name] = {"kind": "numeric", "method": "quantile",
                                    "tie_rule": QUANTILE_TIE_RULE,
                                    "fill": fill,
                                    "reference": sorted_train,
                                    "n_missing": n_missing}
            feature_names.append(spec.name)
            columns_out.append(out)
        else:  # categorical / high_cardinality / redundant-categorical
            raw = [None if v is None else str(v) for v in col]
            train_vals = [raw[i] for i in train_idx if raw[i] is not None]
            counts: dict[str, int] = {}
            for v in train_vals:  # fixed order accumulation
                counts[v] = counts.get(v, 0) + 1
            categories = sorted(counts.keys())
            # mode with tie broken by category code (lexicographic) order
            fill = min(categories, key=lambda c: (-counts[c], c)) if categories else ""
            filled = [fill if v is None else v for v in raw]
            codebook = {c: i for i, c in enumerate(categories)}
            out = [float(codebook.get(v, len(codebook))) for v in filled]
            stats[spec.name] = {"kind": "categorical", "fill": fill,
                                "categories": categories,
                                "n_missing": sum(1 for v in raw if v is None)}
            feature_names.append(spec.name)
            columns_out.append(out)

    matrix = np.array(columns_out, dtype=np.float64).T  # (n_rows, n_features)
    params_fp = fingerprint("preprocess-params", cfg.preprocess_method,
                            cfg.preprocess_params)
    content_fp = fingerprint("preprocess", desc.desc_fingerprint, data_fp,
                             params_fp)
    return {
        "matrix": matrix,
        "feature_names": feature_names,
        "stats": stats,
        "n_train": n_train,
        "method": cfg.preprocess_method,
        "tie_rule": QUANTILE_TIE_RULE if cfg.preprocess_method == "quantile" else None,
        "params_fingerprint": params_fp,
        "content_fingerprint": content_fp,
    }
