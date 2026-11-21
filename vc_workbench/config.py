"""Configuration loading, validation and stable fingerprints."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_INVALID
from .util import sha256_value


class ConfigError(ValueError):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.code = E_CONFIG_INVALID
        self.detail = detail


def _number(raw: dict[str, Any], name: str, default: float | int) -> float | int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    return value


def _integer(raw: dict[str, Any], name: str, default: int) -> int:
    value = raw.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class WorkbenchConfig:
    format_version: int = 1
    sample_rate: int = 16000
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    mel_bands: int = 24
    mel_max_hz: float = 7600.0
    cepstrum_order: int = 12
    f0_min_hz: float = 60.0
    f0_max_hz: float = 400.0
    f0_quantization_levels: int = 64
    thread_pool_workers: int = 4
    speech_frame_budget: int = 1000
    snapshot_interval_segments: int = 1
    max_frames_per_segment: int = 64
    fixed_point_decimals: int = 6
    energy_threshold_db: float = -45.0
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    quantile_interpolation: str = "linear"
    output_gain_db: float = 0.0
    engine_version: int = 1

    @property
    def scale(self) -> int:
        return 10**self.fixed_point_decimals

    @property
    def frame_length_samples(self) -> int:
        return int(round(self.sample_rate * self.frame_length_ms / 1000.0))

    @property
    def frame_shift_samples(self) -> int:
        return int(round(self.sample_rate * self.frame_shift_ms / 1000.0))

    @property
    def energy_threshold_q(self) -> int:
        return int(round(self.energy_threshold_db * self.scale))

    def analysis_fingerprint(self) -> str:
        payload = {
            "engine_version": self.engine_version,
            "sample_rate": self.sample_rate,
            "frame_length_samples": self.frame_length_samples,
            "frame_shift_samples": self.frame_shift_samples,
            "mel_bands": self.mel_bands,
            "mel_max_hz": self.mel_max_hz,
            "cepstrum_order": self.cepstrum_order,
            "f0_min_hz": self.f0_min_hz,
            "f0_max_hz": self.f0_max_hz,
            "f0_quantization_levels": self.f0_quantization_levels,
            "fixed_point_decimals": self.fixed_point_decimals,
            "energy_threshold_db": self.energy_threshold_db,
        }
        return sha256_value(payload)

    def stats_fingerprint(self) -> str:
        payload = {
            "analysis": self.analysis_fingerprint(),
            "quantiles": list(self.quantiles),
            "quantile_interpolation": self.quantile_interpolation,
            "energy_threshold_db": self.energy_threshold_db,
            "fixed_point_decimals": self.fixed_point_decimals,
        }
        return sha256_value(payload)

    def mapping_fingerprint(self) -> str:
        payload = {
            "analysis": self.analysis_fingerprint(),
            "stats": self.stats_fingerprint(),
            "fixed_point_decimals": self.fixed_point_decimals,
            "output_gain_db": self.output_gain_db,
        }
        return sha256_value(payload)

    def assembly_fingerprint(self) -> str:
        return sha256_value(
            {
                "mapping": self.mapping_fingerprint(),
                "sample_rate": self.sample_rate,
                "fixed_point_decimals": self.fixed_point_decimals,
            }
        )

    def config_hash(self) -> str:
        return sha256_value(self.as_public_dict(include_runtime=True))

    def as_public_dict(self, *, include_runtime: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["quantiles"] = list(self.quantiles)
        if not include_runtime:
            for key in ("format_version", "engine_version"):
                data.pop(key, None)
        return data


_ALIASES = {
    "frame_ms": "frame_length_ms",
    "hop_ms": "frame_shift_ms",
    "frame_step_ms": "frame_shift_ms",
    "n_mels": "mel_bands",
    "mel_bands_count": "mel_bands",
    "mel_fmax_hz": "mel_max_hz",
    "cepstral_order": "cepstrum_order",
    "f0_levels": "f0_quantization_levels",
    "workers": "thread_pool_workers",
    "frame_budget": "speech_frame_budget",
    "snapshot_every_segments": "snapshot_interval_segments",
    "max_segment_frames": "max_frames_per_segment",
    "quantization_decimals": "fixed_point_decimals",
    "q_decimals": "fixed_point_decimals",
    "silence_threshold_db": "energy_threshold_db",
    "energy_floor_db": "energy_threshold_db",
    "thread_workers": "thread_pool_workers",
    "snapshot_interval": "snapshot_interval_segments",
}


def load_config(path: Path) -> WorkbenchConfig:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("config root must be an object")
    raw = dict(raw)
    for alias, canonical in _ALIASES.items():
        if alias in raw and canonical not in raw:
            raw[canonical] = raw[alias]
    quantiles = raw.get("quantiles", raw.get("quantile_levels", [0.1, 0.5, 0.9]))
    if not isinstance(quantiles, (list, tuple)) or not quantiles:
        raise ConfigError("quantiles must be a non-empty array")
    parsed_quantiles: list[float] = []
    for value in quantiles:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ConfigError("quantile levels must be numbers in [0, 1]")
        parsed_quantiles.append(float(value))
    interpolation = str(raw.get("quantile_interpolation", raw.get("quantile_method", "linear")))
    if interpolation not in {"linear", "lower", "higher", "midpoint", "nearest"}:
        raise ConfigError(f"unsupported quantile interpolation: {interpolation}")
    cfg = WorkbenchConfig(
        format_version=_integer(raw, "format_version", 1),
        sample_rate=_integer(raw, "sample_rate", 16000),
        frame_length_ms=float(_number(raw, "frame_length_ms", 25.0)),
        frame_shift_ms=float(_number(raw, "frame_shift_ms", 10.0)),
        mel_bands=_integer(raw, "mel_bands", 24),
        mel_max_hz=float(_number(raw, "mel_max_hz", 7600.0)),
        cepstrum_order=_integer(raw, "cepstrum_order", 12),
        f0_min_hz=float(_number(raw, "f0_min_hz", 60.0)),
        f0_max_hz=float(_number(raw, "f0_max_hz", 400.0)),
        f0_quantization_levels=_integer(raw, "f0_quantization_levels", 64),
        thread_pool_workers=_integer(raw, "thread_pool_workers", 4),
        speech_frame_budget=_integer(raw, "speech_frame_budget", 1000),
        snapshot_interval_segments=_integer(raw, "snapshot_interval_segments", 1),
        max_frames_per_segment=_integer(raw, "max_frames_per_segment", 64),
        fixed_point_decimals=_integer(raw, "fixed_point_decimals", 6),
        energy_threshold_db=float(_number(raw, "energy_threshold_db", -45.0)),
        quantiles=tuple(parsed_quantiles),
        quantile_interpolation=interpolation,
        output_gain_db=float(_number(raw, "output_gain_db", 0.0)),
        engine_version=_integer(raw, "engine_version", 1),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: WorkbenchConfig) -> None:
    if cfg.format_version < 1:
        raise ConfigError("format_version must be >= 1")
    if cfg.sample_rate != 16000:
        raise ConfigError("sample_rate must be 16000")
    if not 1 <= cfg.frame_length_ms <= 100:
        raise ConfigError("frame_length_ms must be in (0, 100]")
    if not 1 <= cfg.frame_shift_ms <= cfg.frame_length_ms:
        raise ConfigError("frame_shift_ms must be in [1, frame_length_ms]")
    if cfg.frame_length_samples < 2 or cfg.frame_shift_samples < 1:
        raise ConfigError("frame lengths round to invalid sample counts")
    if cfg.mel_bands < 2:
        raise ConfigError("mel_bands must be >= 2")
    if not 0 < cfg.mel_max_hz <= cfg.sample_rate / 2:
        raise ConfigError("mel_max_hz must be in (0, sample_rate/2]")
    if cfg.cepstrum_order < 1 or cfg.cepstrum_order >= cfg.mel_bands:
        raise ConfigError("cepstrum_order must be in [1, mel_bands)")
    if not 0 < cfg.f0_min_hz < cfg.f0_max_hz:
        raise ConfigError("f0_min_hz must be below f0_max_hz")
    if cfg.f0_max_hz >= cfg.sample_rate / 2:
        raise ConfigError("f0_max_hz must be below Nyquist")
    if cfg.f0_quantization_levels < 2:
        raise ConfigError("f0_quantization_levels must be >= 2")
    if cfg.thread_pool_workers < 1:
        raise ConfigError("thread_pool_workers must be >= 1")
    if cfg.speech_frame_budget < 0:
        raise ConfigError("speech_frame_budget must be >= 0")
    if cfg.snapshot_interval_segments < 1:
        raise ConfigError("snapshot_interval_segments must be >= 1")
    if cfg.max_frames_per_segment < 1:
        raise ConfigError("max_frames_per_segment must be >= 1")
    if not 0 <= cfg.fixed_point_decimals <= 9:
        raise ConfigError("fixed_point_decimals must be in [0, 9]")
    if cfg.output_gain_db < -60 or cfg.output_gain_db > 60:
        raise ConfigError("output_gain_db must be in [-60, 60]")
