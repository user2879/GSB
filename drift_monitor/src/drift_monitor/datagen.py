"""Deterministic synthetic tabular data generation.

A dedicated :class:`numpy.random.Generator` is derived for every
(period, column) pair from the master seed, so generated data is fully
deterministic regardless of how many other columns exist.  Missing values are
kept as explicit masks rather than sentinels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ColumnSpec, DataDesc, SizeTier
from .errors import DataGenError

REF = "reference"
CUR = "current"


@dataclass(frozen=True)
class GeneratedColumn:
    name: str
    kind: str
    values: np.ndarray  # object array; None marks a missing entry
    missing_mask: np.ndarray


@dataclass(frozen=True)
class Dataset:
    period: str
    tier_name: str
    n_samples: int
    columns: tuple[GeneratedColumn, ...]
    data_fingerprint_inputs: dict

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


def _stream_rng(seed: int, period: str, index: int) -> np.random.Generator:
    tag = sum((i + 1) * ord(ch) for i, ch in enumerate(period))
    return np.random.default_rng(np.int64(seed * 1_000_003 + tag * 9176 + index * 37 + 11))


def _missing_rng(seed: int, period: str, index: int) -> np.random.Generator:
    tag = sum((i + 1) * ord(ch) for i, ch in enumerate(period))
    return np.random.default_rng(np.int64(seed * 2_000_003 + tag * 7919 + index * 53 + 5))


def _expanded_columns(desc: DataDesc, tier: SizeTier) -> tuple[ColumnSpec, ...]:
    out: list[ColumnSpec] = []
    numeric_idx = cat_idx = 0
    for base in desc.base_columns:
        if base.kind == "high_cardinality":
            out.append(base)
            continue
        copies = tier.feature_multiplier
        for copy in range(copies):
            suffix = f"__m{copy}" if copies > 1 else ""
            if base.kind == "numeric":
                numeric_idx += 1
                unique = f"{base.name}{suffix}"
            else:
                cat_idx += 1
                unique = f"{base.name}{suffix}"
            out.append(ColumnSpec(unique, base.kind, base.metric, base.drift_scale))
    if len({c.name for c in out}) != len(out):
        raise DataGenError("expanded column names collided", field="data.size_tiers")
    return tuple(out)


def expanded_columns(desc: DataDesc, tier: SizeTier) -> tuple[ColumnSpec, ...]:
    try:
        return _expanded_columns(desc, tier)
    except DataGenError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise DataGenError(str(exc), field="data") from exc


def _generate_numeric(spec: ColumnSpec, rng: np.random.Generator,
                      n: int, period: str) -> np.ndarray:
    loc = float(abs(hash_index(spec.name)) % 100) / 10.0
    scale = 1.0 + (abs(hash_index(spec.name + "_s")) % 50) / 100.0
    if period == CUR:
        loc = loc + spec.drift_scale * scale
        scale = scale * (1.0 + 0.15 * spec.drift_scale)
    return rng.normal(loc, scale, n)


def hash_index(text: str) -> int:
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) & 0xFFFFFFFF
    return value - 0x80000000


def _generate_categorical(spec: ColumnSpec, rng: np.random.Generator,
                          n: int, period: str) -> list[str]:
    k = 4 + abs(hash_index(spec.name + "_k")) % 4
    vocab = [f"{spec.name}_v{idx}" for idx in range(k)]
    weights = np.array([1.0 / (idx + 1) for idx in range(k)])
    if period == CUR and spec.drift_scale > 0:
        shift = spec.drift_scale * 0.25
        weights = weights * np.array([1.0 + shift * (1 if idx % 2 == 0 else -1)
                                      for idx in range(k)])
    weights = weights / weights.sum()
    draws = rng.choice(k, size=n, p=weights)
    return [vocab[idx] for idx in draws]


def _generate_high_card(spec: ColumnSpec, rng: np.random.Generator,
                        n: int, period: str) -> list[str]:
    values = []
    salt = 1 if period == CUR else 0
    for idx in range(n):
        bucket = int(rng.integers(0, max(2, n // 4)))
        if period == CUR and spec.drift_scale > 0 and rng.random() < 0.2 * spec.drift_scale:
            bucket = bucket + 10_000
        values.append(f"hc_{spec.name}_{bucket + salt}")
    return values


def generate_period(desc: DataDesc, tier: SizeTier, period: str) -> Dataset:
    if period not in (REF, CUR):
        raise DataGenError(f"unknown period {period}", field="period")
    try:
        columns_spec = expanded_columns(desc, tier)
        generated = []
        for index, spec in enumerate(columns_spec):
            rng = _stream_rng(desc.seed, period, index)
            miss_rng = _missing_rng(desc.seed, period, index)
            n = tier.samples
            if spec.kind == "numeric":
                raw = _generate_numeric(spec, rng, n, period)
            elif spec.kind == "categorical":
                raw = _generate_categorical(spec, rng, n, period)
            else:
                raw = _generate_high_card(spec, rng, n, period)
            values = np.empty(n, dtype=object)
            mask = miss_rng.random(n) < desc.missing_rate
            for idx in range(n):
                values[idx] = None if mask[idx] else raw[idx]
            generated.append(GeneratedColumn(spec.name, spec.kind, values, mask))
        return Dataset(
            period=period,
            tier_name=tier.name,
            n_samples=tier.samples,
            columns=tuple(generated),
            data_fingerprint_inputs={
                "seed": desc.seed,
                "missing_rate": desc.missing_rate,
                "tier": tier.name,
                "samples": tier.samples,
                "feature_multiplier": tier.feature_multiplier,
                "period": period,
                "columns": [c.name for c in columns_spec],
            },
        )
    except DataGenError:
        raise
    except Exception as exc:
        raise DataGenError(str(exc), field="data") from exc


def generate_pair(desc: DataDesc, tier: SizeTier) -> tuple[Dataset, Dataset]:
    return generate_period(desc, tier, REF), generate_period(desc, tier, CUR)
