from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import SchemaError
from .util import json_hash, load_json


def _require_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{where} must be an integer", where=where)
    if minimum is not None and value < minimum:
        raise SchemaError(f"{where} must be >= {minimum}", where=where)
    return value


def _require_number(
    value: Any, where: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{where} must be a number", where=where)
    number = float(value)
    if not math.isfinite(number):
        raise SchemaError(f"{where} must be finite", where=where)
    if minimum is not None and number < minimum:
        raise SchemaError(f"{where} must be >= {minimum}", where=where)
    if maximum is not None and number > maximum:
        raise SchemaError(f"{where} must be <= {maximum}", where=where)
    return number


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{where} must be an object", where=where)
    return value


@dataclass(frozen=True)
class EndpointConfig:
    frame_length: int
    frame_shift: int
    high_energy: float
    low_energy: float
    min_speech_ms: int
    min_silence_ms: int

    @classmethod
    def from_dict(cls, value: Any) -> "EndpointConfig":
        item = _mapping(value, "config.endpoint")
        result = cls(
            frame_length=_require_int(item.get("frame_length"), "endpoint.frame_length", minimum=1),
            frame_shift=_require_int(item.get("frame_shift"), "endpoint.frame_shift", minimum=1),
            high_energy=_require_number(item.get("high_energy"), "endpoint.high_energy", minimum=0.0),
            low_energy=_require_number(item.get("low_energy"), "endpoint.low_energy", minimum=0.0),
            min_speech_ms=_require_int(item.get("min_speech_ms"), "endpoint.min_speech_ms", minimum=1),
            min_silence_ms=_require_int(item.get("min_silence_ms"), "endpoint.min_silence_ms", minimum=1),
        )
        if result.high_energy <= result.low_energy:
            raise SchemaError("endpoint.high_energy must be greater than low_energy", where="endpoint")
        if result.frame_shift > result.frame_length:
            raise SchemaError("endpoint.frame_shift must not exceed frame_length", where="endpoint")
        return result


@dataclass(frozen=True)
class AnalysisConfig:
    window_length: int
    window_shift: int
    frame_length: int
    frame_shift: int
    mel_bands: int
    quant_scale: int
    frac_bits: int
    allowed_quant_scales: tuple[int, ...]
    allowed_frac_bits: tuple[int, ...]
    posterior_quantum_ppm: int

    @classmethod
    def from_dict(cls, value: Any) -> "AnalysisConfig":
        item = _mapping(value, "config.analysis")
        allowed_scales_raw = item.get("allowed_quant_scales")
        allowed_bits_raw = item.get("allowed_frac_bits")
        if not isinstance(allowed_scales_raw, list) or not allowed_scales_raw:
            raise SchemaError("analysis.allowed_quant_scales must be a non-empty array", where="analysis")
        if not isinstance(allowed_bits_raw, list) or not allowed_bits_raw:
            raise SchemaError("analysis.allowed_frac_bits must be a non-empty array", where="analysis")
        allowed_scales = tuple(
            _require_int(scale, "analysis.allowed_quant_scales[]", minimum=1)
            for scale in allowed_scales_raw
        )
        allowed_bits = tuple(
            _require_int(bits, "analysis.allowed_frac_bits[]", minimum=0) for bits in allowed_bits_raw
        )
        result = cls(
            window_length=_require_int(item.get("window_length"), "analysis.window_length", minimum=1),
            window_shift=_require_int(item.get("window_shift"), "analysis.window_shift", minimum=1),
            frame_length=_require_int(item.get("frame_length"), "analysis.frame_length", minimum=1),
            frame_shift=_require_int(item.get("frame_shift"), "analysis.frame_shift", minimum=1),
            mel_bands=_require_int(item.get("mel_bands"), "analysis.mel_bands", minimum=1),
            quant_scale=_require_int(item.get("quant_scale"), "analysis.quant_scale", minimum=1),
            frac_bits=_require_int(item.get("frac_bits"), "analysis.frac_bits", minimum=0),
            allowed_quant_scales=allowed_scales,
            allowed_frac_bits=allowed_bits,
            posterior_quantum_ppm=_require_int(
                item.get("posterior_quantum_ppm"), "analysis.posterior_quantum_ppm", minimum=1
            ),
        )
        if len(set(allowed_scales)) != len(allowed_scales):
            raise SchemaError("analysis.allowed_quant_scales contains duplicates", where="analysis")
        if len(set(allowed_bits)) != len(allowed_bits):
            raise SchemaError("analysis.allowed_frac_bits contains duplicates", where="analysis")
        if result.window_length < result.frame_length:
            raise SchemaError("analysis.window_length must be >= frame_length", where="analysis")
        if result.window_shift > result.window_length:
            raise SchemaError("analysis.window_shift must not exceed window_length", where="analysis")
        if result.frame_shift > result.frame_length:
            raise SchemaError("analysis.frame_shift must not exceed frame_length", where="analysis")
        if result.quant_scale not in result.allowed_quant_scales:
            raise SchemaError(
                "analysis.quant_scale is not in allowed_quant_scales", where="analysis.quant_scale"
            )
        if result.frac_bits not in result.allowed_frac_bits:
            raise SchemaError(
                "analysis.frac_bits is not in allowed_frac_bits", where="analysis.frac_bits"
            )
        return result

    @property
    def quant_profile(self) -> str:
        return f"q{self.frac_bits}_s{self.quant_scale}"

    @property
    def fixed_denominator(self) -> int:
        return self.quant_scale * (1 << self.frac_bits)

    def as_parameters(self) -> dict[str, Any]:
        return {
            "window_length": self.window_length,
            "window_shift": self.window_shift,
            "frame_length": self.frame_length,
            "frame_shift": self.frame_shift,
            "mel_bands": self.mel_bands,
            "quant_scale": self.quant_scale,
            "frac_bits": self.frac_bits,
            "quant_profile": self.quant_profile,
            "posterior_quantum_ppm": self.posterior_quantum_ppm,
        }


@dataclass(frozen=True)
class AdjudicationConfig:
    frame_endpoint_threshold_ppm: int
    min_command_ms: int
    tie_half_step_ppm: int
    min_overlap_frames: int

    @classmethod
    def from_dict(cls, value: Any) -> "AdjudicationConfig":
        item = _mapping(value, "config.adjudication")
        result = cls(
            frame_endpoint_threshold_ppm=_require_int(
                item.get("frame_endpoint_threshold_ppm"),
                "adjudication.frame_endpoint_threshold_ppm",
                minimum=0,
            ),
            min_command_ms=_require_int(
                item.get("min_command_ms"), "adjudication.min_command_ms", minimum=1
            ),
            tie_half_step_ppm=_require_int(
                item.get("tie_half_step_ppm"), "adjudication.tie_half_step_ppm", minimum=0
            ),
            min_overlap_frames=_require_int(
                item.get("min_overlap_frames"), "adjudication.min_overlap_frames", minimum=1
            ),
        )
        if result.frame_endpoint_threshold_ppm > 1_000_000:
            raise SchemaError(
                "adjudication.frame_endpoint_threshold_ppm must be <= 1000000",
                where="adjudication",
            )
        return result

    def as_parameters(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SchedulerConfig:
    max_workers: int
    point_budget: int
    per_interval_quota: int
    point_cost: int

    @classmethod
    def from_dict(cls, value: Any) -> "SchedulerConfig":
        item = _mapping(value, "config.scheduler")
        result = cls(
            max_workers=_require_int(item.get("max_workers"), "scheduler.max_workers", minimum=1),
            point_budget=_require_int(item.get("point_budget"), "scheduler.point_budget", minimum=1),
            per_interval_quota=_require_int(
                item.get("per_interval_quota"), "scheduler.per_interval_quota", minimum=1
            ),
            point_cost=_require_int(item.get("point_cost"), "scheduler.point_cost", minimum=1),
        )
        if result.max_workers > 64:
            raise SchemaError("scheduler.max_workers must be <= 64", where="scheduler")
        return result


@dataclass(frozen=True)
class ReceiptConfig:
    field_widths: dict[str, int]
    quant_schemes: tuple[str, ...]
    default_quant_scheme: str

    @classmethod
    def from_dict(cls, value: Any) -> "ReceiptConfig":
        item = _mapping(value, "config.receipt")
        widths = _mapping(item.get("field_widths"), "receipt.field_widths")
        required = ("interval_id", "speaker_id", "confidence", "command_code")
        parsed: dict[str, int] = {}
        for name in required:
            parsed[name] = _require_int(widths.get(name), f"receipt.field_widths.{name}", minimum=1)
        schemes_raw = item.get("quant_schemes")
        if not isinstance(schemes_raw, list) or not schemes_raw:
            raise SchemaError("receipt.quant_schemes must be a non-empty array", where="receipt")
        schemes: list[str] = []
        for scheme in schemes_raw:
            if not isinstance(scheme, str) or not scheme:
                raise SchemaError("receipt.quant_schemes[] must be a non-empty string", where="receipt")
            schemes.append(scheme)
        if len(set(schemes)) != len(schemes):
            raise SchemaError("receipt.quant_schemes contains duplicates", where="receipt")
        default = item.get("default_quant_scheme")
        if default not in schemes:
            raise SchemaError(
                "receipt.default_quant_scheme must be one of quant_schemes", where="receipt"
            )
        return cls(field_widths=parsed, quant_schemes=tuple(schemes), default_quant_scheme=str(default))


@dataclass(frozen=True)
class DiagnosticConfig:
    schema_version: int
    global_version: int
    seed: int
    sample_rate: int
    channels: int
    bits_per_sample: int
    endpoint: EndpointConfig
    analysis: AnalysisConfig
    adjudication: AdjudicationConfig
    scheduler: SchedulerConfig
    receipt: ReceiptConfig

    @classmethod
    def from_dict(cls, value: Any) -> "DiagnosticConfig":
        item = _mapping(value, "config")
        result = cls(
            schema_version=_require_int(item.get("schema_version"), "config.schema_version", minimum=1),
            global_version=_require_int(item.get("global_version"), "config.global_version", minimum=1),
            seed=_require_int(item.get("seed"), "config.seed"),
            sample_rate=_require_int(item.get("sample_rate"), "config.sample_rate", minimum=1),
            channels=_require_int(item.get("channels"), "config.channels", minimum=1),
            bits_per_sample=_require_int(item.get("bits_per_sample"), "config.bits_per_sample", minimum=1),
            endpoint=EndpointConfig.from_dict(item.get("endpoint")),
            analysis=AnalysisConfig.from_dict(item.get("analysis")),
            adjudication=AdjudicationConfig.from_dict(item.get("adjudication")),
            scheduler=SchedulerConfig.from_dict(item.get("scheduler")),
            receipt=ReceiptConfig.from_dict(item.get("receipt")),
        )
        if result.schema_version != 1:
            raise SchemaError("unsupported config.schema_version", where="config.schema_version")
        if result.sample_rate != 16000:
            raise SchemaError("sample_rate must be 16000", where="config.sample_rate")
        if result.channels != 1:
            raise SchemaError("channels must be 1", where="config.channels")
        if result.bits_per_sample != 16:
            raise SchemaError("bits_per_sample must be 16", where="config.bits_per_sample")
        return result

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        return json_hash(self.as_dict())

    @property
    def projection_seed_material(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "seed": self.seed,
            "quant_profile": self.analysis.quant_profile,
            "mel_bands": self.analysis.mel_bands,
        }


def load_config(path: str | Path) -> DiagnosticConfig:
    config_path = Path(path)
    try:
        raw = load_json(config_path)
    except FileNotFoundError as exc:
        raise SchemaError("config file does not exist", where=str(config_path)) from exc
    except Exception as exc:
        raise SchemaError(f"cannot parse config JSON: {exc}", where=str(config_path)) from exc
    return DiagnosticConfig.from_dict(raw)
