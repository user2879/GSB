"""Deterministic local synthesis of multi-channel industrial vibration blocks.

The generator has no external inputs.  A fixed integer seed plus stable block
and channel counters drive NumPy's PCG64 generator, so every byte in the sample
stream can be reproduced offline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import is_int, sha256_value
from .config import AppConfig
from .errors import E_CONFIG_INVALID, ValidationError


@dataclass(frozen=True)
class SynthSpec:
    seed: int
    block_count: int
    samples_per_block_channel0: int
    acquisition_step_ms: int
    trend_slope: float
    impulse_period_ms: float
    impulse_amplitude: float
    impulse_decay: float
    noise_amplitude: float
    regime_period_blocks: int
    missing_every: int
    missing_start: int
    missing_length: int
    resampling_every: int
    resampling_start: int
    resampling_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "block_count": self.block_count,
            "samples_per_block_channel0": self.samples_per_block_channel0,
            "acquisition_step_ms": self.acquisition_step_ms,
            "trend_slope": self.trend_slope,
            "impulse_period_ms": self.impulse_period_ms,
            "impulse_amplitude": self.impulse_amplitude,
            "impulse_decay": self.impulse_decay,
            "noise_amplitude": self.noise_amplitude,
            "regime_period_blocks": self.regime_period_blocks,
            "missing_every": self.missing_every,
            "missing_start": self.missing_start,
            "missing_length": self.missing_length,
            "resampling_every": self.resampling_every,
            "resampling_start": self.resampling_start,
            "resampling_length": self.resampling_length,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_value({"kind": "synth-spec", "value": self.to_dict()})


def _number(obj: dict[str, Any], key: str, default: float) -> float:
    value = obj.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(E_CONFIG_INVALID, "must be a finite number", key)
    return float(value)


def _integer(obj: dict[str, Any], key: str, default: int, minimum: int = 0) -> int:
    value = obj.get(key, default)
    if not is_int(value) or value < minimum:
        raise ValidationError(E_CONFIG_INVALID, f"must be an integer >= {minimum}", key)
    return value


def parse_synth_spec(data: Any) -> SynthSpec:
    if not isinstance(data, dict):
        raise ValidationError(E_CONFIG_INVALID, "synth spec must be an object", "$")
    return SynthSpec(
        seed=_integer(data, "seed", 20261127, 0),
        block_count=_integer(data, "block_count", 16, 1),
        samples_per_block_channel0=_integer(data, "samples_per_block_channel0", 256, 1),
        acquisition_step_ms=_integer(data, "acquisition_step_ms", 128, 1),
        trend_slope=_number(data, "trend_slope", 0.0008),
        impulse_period_ms=_number(data, "impulse_period_ms", 37.0),
        impulse_amplitude=_number(data, "impulse_amplitude", 1.8),
        impulse_decay=_number(data, "impulse_decay", 80.0),
        noise_amplitude=_number(data, "noise_amplitude", 0.12),
        regime_period_blocks=_integer(data, "regime_period_blocks", 5, 1),
        missing_every=_integer(data, "missing_every", 5, 1),
        missing_start=_integer(data, "missing_start", 48, 0),
        missing_length=_integer(data, "missing_length", 17, 1),
        resampling_every=_integer(data, "resampling_every", 4, 1),
        resampling_start=_integer(data, "resampling_start", 160, 0),
        resampling_length=_integer(data, "resampling_length", 24, 1),
    )


def load_synth_spec(path: str | Path) -> SynthSpec:
    with Path(path).open("r", encoding="utf-8") as handle:
        return parse_synth_spec(json.load(handle))


def _synthesize_channel(
    *,
    seed: int,
    block_index: int,
    channel_index: int,
    sample_rate_hz: int,
    acquisition_ms: int,
    sample_count: int,
    spec: SynthSpec,
) -> list[float]:
    block_rng = np.random.Generator(
        np.random.PCG64(seed + block_index * 1009 + channel_index * 104_729)
    )
    start_index = (acquisition_ms * sample_rate_hz) // 1000
    local_index = np.arange(sample_count, dtype=np.int64)
    global_index = start_index + local_index
    seconds = global_index.astype(np.float64) / sample_rate_hz

    regime = (block_index // spec.regime_period_blocks) % 3
    regime_gain = (0.82, 1.0, 1.19)[regime]
    trend = spec.trend_slope * seconds * (1.0 + channel_index * 0.13)

    period_seconds = spec.impulse_period_ms / 1000.0
    phase_seconds = np.mod(seconds, period_seconds)
    impulses = np.where(
        phase_seconds < min(0.008, period_seconds / 3.0),
        np.exp(-phase_seconds * spec.impulse_decay),
        0.0,
    )
    impulses = impulses * spec.impulse_amplitude * regime_gain
    tone = 0.22 * np.sin(2.0 * np.pi * (43.0 + 7.0 * channel_index) * seconds)
    noise = block_rng.normal(0.0, spec.noise_amplitude, size=sample_count)
    values = trend + tone + impulses + noise
    if channel_index == 1:
        values = values * 0.93 + 0.06 * np.sin(2.0 * np.pi * 71.0 * seconds)
    return [round(float(value), 12) for value in values]


def generate_blocks(config: AppConfig, spec: SynthSpec) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block_index in range(spec.block_count):
        acquisition_ms = block_index * spec.acquisition_step_ms
        sequence = block_index + 1
        signals: dict[str, list[float]] = {}
        missing_segments: list[dict[str, Any]] = []
        resampling_markers: list[dict[str, Any]] = []
        for channel_index, channel in enumerate(config.channels):
            if channel_index == 0:
                sample_count = spec.samples_per_block_channel0
            else:
                duration_ms = spec.samples_per_block_channel0 * 1000 / config.channels[0].sample_rate_hz
                sample_count = int(round(duration_ms * channel.sample_rate_hz / 1000))
            signals[channel.channel_id] = _synthesize_channel(
                seed=spec.seed,
                block_index=block_index,
                channel_index=channel_index,
                sample_rate_hz=channel.sample_rate_hz,
                acquisition_ms=acquisition_ms,
                sample_count=sample_count,
                spec=spec,
            )
            if spec.missing_every > 0 and block_index % spec.missing_every == spec.missing_every - 1:
                start = min(spec.missing_start, max(0, sample_count - 1))
                end = min(sample_count, start + spec.missing_length)
                if end > start:
                    missing_segments.append(
                        {"channel": channel.channel_id, "start": start, "end": end}
                    )
            if spec.resampling_every > 0 and block_index % spec.resampling_every == 1:
                start = min(spec.resampling_start, max(0, sample_count - 1))
                end = min(sample_count, start + spec.resampling_length)
                if end > start:
                    source_rate = channel.sample_rate_hz
                    target_rate = max(1, source_rate - 125)
                    resampling_markers.append(
                        {
                            "channel": channel.channel_id,
                            "start": start,
                            "end": end,
                            "source_rate_hz": source_rate,
                            "target_rate_hz": target_rate,
                        }
                    )
        blocks.append(
            {
                "block_id": f"b{sequence:04d}",
                "arrival_seq": sequence,
                "acquisition_ms": acquisition_ms,
                "signals": signals,
                "missing_segments": missing_segments,
                "resampling_markers": resampling_markers,
            }
        )
    return blocks


def write_generated_stream(config: AppConfig, spec: SynthSpec, output: str | Path) -> dict[str, Any]:
    blocks = generate_blocks(config, spec)
    document = {
        "schema_version": 1,
        "generator": spec.to_dict(),
        "generator_fingerprint": spec.fingerprint,
        "blocks": blocks,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return document
