"""Evaluation configuration and stage-specific fingerprints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

from .errors import FormatError
from .util import decimal_fraction, read_json, sha256_json


DIMENSIONS = ("level", "noise", "clipping", "continuity")
HARD_CONCURRENCY_CAP = 16


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise FormatError(f"{name} must be numeric")
    return float(value)


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FormatError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class Config:
    sample_rate: int
    frame_length_ms: int
    hop_ms: int
    silence_threshold_dbfs: float
    noise_floor_percentile: int
    clipping_amplitude_threshold: float
    consecutive_zero_samples: int
    weights: dict[str, float]
    fixed_decimal_bits: int
    quantization_levels: tuple[int, ...]
    max_concurrency: int
    concurrency_cap: int
    snapshot_commit_interval: int
    verify_serial_consistency: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        if not isinstance(raw, dict):
            raise FormatError("configuration must be a JSON object")
        required = {
            "sample_rate",
            "frame_length_ms",
            "hop_ms",
            "silence_threshold_dbfs",
            "noise_floor_percentile",
            "clipping_amplitude_threshold",
            "consecutive_zero_samples",
            "weights",
            "fixed_decimal_bits",
            "quantization_levels",
            "max_concurrency",
            "snapshot_commit_interval",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise FormatError(f"missing configuration fields: {', '.join(missing)}")
        sample_rate = _require_int(raw["sample_rate"], "sample_rate")
        frame_length_ms = _require_int(raw["frame_length_ms"], "frame_length_ms")
        hop_ms = _require_int(raw["hop_ms"], "hop_ms")
        silence = _require_number(raw["silence_threshold_dbfs"], "silence_threshold_dbfs")
        percentile = _require_int(raw["noise_floor_percentile"], "noise_floor_percentile")
        clip_threshold = _require_number(
            raw["clipping_amplitude_threshold"], "clipping_amplitude_threshold"
        )
        zero_samples = _require_int(raw["consecutive_zero_samples"], "consecutive_zero_samples")
        fixed_bits = _require_int(raw["fixed_decimal_bits"], "fixed_decimal_bits")
        max_concurrency = _require_int(raw["max_concurrency"], "max_concurrency")
        cap = _require_int(raw.get("concurrency_cap", HARD_CONCURRENCY_CAP), "concurrency_cap")
        interval = _require_int(raw["snapshot_commit_interval"], "snapshot_commit_interval")
        verify_serial = bool(raw.get("verify_serial_consistency", True))
        weights_raw = raw["weights"]
        if not isinstance(weights_raw, dict) or set(weights_raw) != set(DIMENSIONS):
            raise FormatError(f"weights must contain exactly {', '.join(DIMENSIONS)}")
        weights: dict[str, float] = {}
        for dim in DIMENSIONS:
            weight = _require_number(weights_raw[dim], f"weights.{dim}")
            if weight <= 0:
                raise FormatError(f"weights.{dim} must be positive")
            weights[dim] = weight
        quantization = raw["quantization_levels"]
        if not isinstance(quantization, list) or not quantization:
            raise FormatError("quantization_levels must be a non-empty array")
        levels: list[int] = []
        for item in quantization:
            level = _require_int(item, "quantization_levels item")
            if not 0 <= level <= 1000:
                raise FormatError("quantization_levels values must be in 0..1000")
            levels.append(level)
        if levels != sorted(set(levels)):
            raise FormatError("quantization_levels must be sorted and unique")
        if sample_rate <= 0 or frame_length_ms <= 0 or hop_ms <= 0:
            raise FormatError("sample rate and frame values must be positive")
        if hop_ms > frame_length_ms:
            raise FormatError("hop_ms must not exceed frame_length_ms")
        if not -160 <= silence <= 0:
            raise FormatError("silence_threshold_dbfs must be in -160..0")
        if not 1 <= percentile <= 100:
            raise FormatError("noise_floor_percentile must be in 1..100")
        if not 0 < clip_threshold <= 1:
            raise FormatError("clipping_amplitude_threshold must be in (0, 1]")
        if zero_samples < 1:
            raise FormatError("consecutive_zero_samples must be positive")
        if not 0 <= fixed_bits <= 6:
            raise FormatError("fixed_decimal_bits must be in 0..6")
        if max_concurrency < 1 or cap < 1 or cap > HARD_CONCURRENCY_CAP:
            raise FormatError(f"concurrency cap must be in 1..{HARD_CONCURRENCY_CAP}")
        if interval < 1:
            raise FormatError("snapshot_commit_interval must be positive")
        return cls(
            sample_rate=sample_rate,
            frame_length_ms=frame_length_ms,
            hop_ms=hop_ms,
            silence_threshold_dbfs=silence,
            noise_floor_percentile=percentile,
            clipping_amplitude_threshold=clip_threshold,
            consecutive_zero_samples=zero_samples,
            weights=weights,
            fixed_decimal_bits=fixed_bits,
            quantization_levels=tuple(levels),
            max_concurrency=max_concurrency,
            concurrency_cap=cap,
            snapshot_commit_interval=interval,
            verify_serial_consistency=verify_serial,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        try:
            raw = read_json(path)
        except (OSError, ValueError) as exc:
            raise FormatError(f"cannot read configuration: {exc}") from exc
        return cls.from_dict(raw)

    @property
    def frame_length_samples(self) -> int:
        return int(round(self.sample_rate * self.frame_length_ms / 1000))

    @property
    def hop_samples(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000))

    @property
    def worker_count(self) -> int:
        return min(self.max_concurrency, self.concurrency_cap, HARD_CONCURRENCY_CAP)

    @property
    def fraction_weights(self) -> dict[str, Fraction]:
        return {name: decimal_fraction(Decimal(str(value))) for name, value in self.weights.items()}

    @property
    def config_hash(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def frame_fingerprint(self) -> str:
        return sha256_json(
            {
                "sample_rate": self.sample_rate,
                "frame_length_samples": self.frame_length_samples,
                "hop_samples": self.hop_samples,
                "clipping_amplitude_threshold": self.clipping_amplitude_threshold,
                "silence_threshold_dbfs": self.silence_threshold_dbfs,
                "noise_floor_percentile": self.noise_floor_percentile,
                "consecutive_zero_samples": self.consecutive_zero_samples,
                "extractor": "features-summary-v1",
            }
        )

    @property
    def threshold_fingerprint(self) -> str:
        return sha256_json(
            {
                "silence_threshold_dbfs": self.silence_threshold_dbfs,
                "noise_floor_percentile": self.noise_floor_percentile,
                "clipping_amplitude_threshold": self.clipping_amplitude_threshold,
                "consecutive_zero_samples": self.consecutive_zero_samples,
                "quantization_levels": self.quantization_levels,
                "scorer": "dimension-bands-v1",
            }
        )

    @property
    def weights_fingerprint(self) -> str:
        return sha256_json(
            {
                "weights": {key: format(value, ".17g") for key, value in sorted(self.weights.items())},
                "combiner": "weighted-fixed-v1",
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "frame_length_ms": self.frame_length_ms,
            "hop_ms": self.hop_ms,
            "silence_threshold_dbfs": self.silence_threshold_dbfs,
            "noise_floor_percentile": self.noise_floor_percentile,
            "clipping_amplitude_threshold": self.clipping_amplitude_threshold,
            "consecutive_zero_samples": self.consecutive_zero_samples,
            "weights": dict(self.weights),
            "fixed_decimal_bits": self.fixed_decimal_bits,
            "quantization_levels": list(self.quantization_levels),
            "max_concurrency": self.max_concurrency,
            "concurrency_cap": self.concurrency_cap,
            "snapshot_commit_interval": self.snapshot_commit_interval,
            "verify_serial_consistency": self.verify_serial_consistency,
        }

    def with_updates(self, **updates: Any) -> "Config":
        raw = self.to_dict()
        raw.update(updates)
        return Config.from_dict(raw)


FILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,23}$")
LEVEL_CODE_PATTERN = re.compile(r"^[A-Za-z0-9]{1,2}$")
LEVEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
