from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import E_CONFIG_INVALID, E_QUANTIZATION_LEVEL, BusinessError


ALLOWED_QUANTILE_METHODS = {"linear", "lower", "higher", "midpoint", "nearest"}
ALLOWED_METRICS = {
    "f0_mean_hz",
    "f0_std_hz",
    "f0_median_hz",
    "f0_p10_hz",
    "f0_p90_hz",
    "f1_mean_hz",
    "f2_mean_hz",
    "speech_rate_sps",
    "voiced_ratio",
    "rms_db",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DimensionConfig:
    name: str
    metric: str
    dynamic_range: tuple[float, float]
    target_step: float
    bit_widths: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "dynamic_range": list(self.dynamic_range),
            "target_step": self.target_step,
            "bit_widths": list(self.bit_widths),
        }


@dataclass
class AnalysisConfig:
    raw: dict[str, Any]
    source_path: Path | None = None
    sample_rate: int = 16000
    channels: int = 1
    sample_width_bits: int = 16
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    energy_enter_db: float = -42.0
    energy_exit_db: float = -55.0
    min_segment_ms: float = 90.0
    min_gap_ms: float = 60.0
    quantile_interpolation: str = "linear"
    f0_min_hz: float = 60.0
    f0_max_hz: float = 500.0
    voiced_autocorr: float = 0.28
    lpc_order: int = 16
    f1_range_hz: tuple[float, float] = (180.0, 1100.0)
    f2_range_hz: tuple[float, float] = (700.0, 3500.0)
    style_rate_bins_sps: tuple[float, float] = (3.8, 5.0)
    style_labels: tuple[str, str, str] = ("slow", "neutral", "fast")
    high_variation_f0_std_hz: float = 45.0
    high_variation_label: str = "expressive"
    dimensions: tuple[DimensionConfig, ...] = ()
    similarity_threshold: float = 0.86
    similarity_weights: dict[str, float] = field(default_factory=dict)
    preview_budget_ms: float = 250.0
    max_workers: int = 4
    snapshot_commit_every: int = 1
    verify_incremental_vs_full: bool = True
    initial_annotations: tuple[dict[str, Any], ...] = ()
    schema_version: int = 1

    @property
    def frame_length_samples(self) -> int:
        return max(1, int(round(self.frame_length_ms * self.sample_rate / 1000.0)))

    @property
    def frame_shift_samples(self) -> int:
        return max(1, int(round(self.frame_shift_ms * self.sample_rate / 1000.0)))

    @property
    def config_hash(self) -> str:
        return digest_json(self.raw)

    def dimension_names(self) -> list[str]:
        return [dimension.name for dimension in self.dimensions]

    def selected_bits(self, dimension_name: str, overrides: dict[str, int] | None = None) -> int:
        overrides = overrides or {}
        if dimension_name in overrides:
            return self.validate_bit_width(dimension_name, int(overrides[dimension_name]))
        dimension = next(item for item in self.dimensions if item.name == dimension_name)
        low, high = dimension.dynamic_range
        span = high - low
        for bits in sorted(dimension.bit_widths):
            level_count = (1 << bits) - 1
            step = span / level_count if level_count else math.inf
            if step <= dimension.target_step:
                return bits
        return max(dimension.bit_widths)

    def validate_bit_width(self, dimension_name: str, bits: int) -> int:
        dimension = next((item for item in self.dimensions if item.name == dimension_name), None)
        if dimension is None:
            raise BusinessError(E_QUANTIZATION_LEVEL, f"unknown timbre dimension: {dimension_name}")
        if bits not in dimension.bit_widths:
            raise BusinessError(
                E_QUANTIZATION_LEVEL,
                f"bit width {bits} is not allowed for {dimension_name}",
                dimension=dimension_name,
                bit_width=bits,
                allowed=list(dimension.bit_widths),
            )
        return bits

    def quantization_plan(self, overrides: dict[str, int] | None = None) -> list[dict[str, Any]]:
        plan: list[dict[str, Any]] = []
        for dimension in self.dimensions:
            bits = self.selected_bits(dimension.name, overrides)
            low, high = dimension.dynamic_range
            step = (high - low) / ((1 << bits) - 1)
            plan.append(
                {
                    "name": dimension.name,
                    "metric": dimension.metric,
                    "bit_width": bits,
                    "step": step,
                    "range": [low, high],
                    "source": "override" if overrides and dimension.name in overrides else "adaptive",
                    "allowed_bit_widths": list(dimension.bit_widths),
                }
            )
        return plan

    def quantization_fingerprint(self, overrides: dict[str, int] | None = None) -> str:
        return digest_json(
            {
                "plan": self.quantization_plan(overrides),
                "overrides": overrides or {},
                "dimensions": [dimension.as_dict() for dimension in self.dimensions],
            }
        )

    def feature_fingerprint(self) -> str:
        payload = {
            key: self.raw.get(key)
            for key in (
                "sample_rate",
                "channels",
                "sample_width_bits",
                "frame_length_ms",
                "frame_shift_ms",
                "energy",
                "min_segment_ms",
                "min_gap_ms",
                "quantile_interpolation",
                "f0",
                "formants",
                "style",
            )
        }
        return digest_json(payload)

    def assignment_fingerprint(self) -> str:
        return digest_json({"similarity": self.raw.get("similarity"), "dimensions": self.dimension_names()})


def _number(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BusinessError(E_CONFIG_INVALID, f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise BusinessError(E_CONFIG_INVALID, f"{field_name} must be finite")
    return number


def _integer(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BusinessError(E_CONFIG_INVALID, f"{field_name} must be an integer") from exc


def _pair(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise BusinessError(E_CONFIG_INVALID, f"{field_name} must be a two-element array")
    return (_number(value[0], f"{field_name}[0]"), _number(value[1], f"{field_name}[1]"))


def parse_config(raw: dict[str, Any], source_path: Path | None = None) -> AnalysisConfig:
    if not isinstance(raw, dict):
        raise BusinessError(E_CONFIG_INVALID, "configuration root must be an object")
    schema_version = _integer(raw.get("schema_version", 1), "schema_version")
    if schema_version != 1:
        raise BusinessError(E_CONFIG_INVALID, "unsupported configuration schema version", schema_version=schema_version)
    sample_rate = _integer(raw.get("sample_rate", 16000), "sample_rate")
    channels = _integer(raw.get("channels", 1), "channels")
    width = _integer(raw.get("sample_width_bits", 16), "sample_width_bits")
    if (sample_rate, channels, width) != (16000, 1, 16):
        raise BusinessError(
            E_CONFIG_INVALID,
            "this build requires 16 kHz, mono, 16-bit PCM",
            sample_rate=sample_rate,
            channels=channels,
            sample_width_bits=width,
        )
    frame_length_ms = _number(raw.get("frame_length_ms"), "frame_length_ms")
    frame_shift_ms = _number(raw.get("frame_shift_ms"), "frame_shift_ms")
    if frame_length_ms <= 0 or frame_shift_ms <= 0:
        raise BusinessError(E_CONFIG_INVALID, "frame length and shift must be positive")
    energy = raw.get("energy", {})
    enter_db = _number(energy.get("enter_db", -42.0), "energy.enter_db")
    exit_db = _number(energy.get("exit_db", -55.0), "energy.exit_db")
    if enter_db <= exit_db:
        raise BusinessError(E_CONFIG_INVALID, "energy.enter_db must be greater than energy.exit_db")
    min_segment_ms = _number(raw.get("min_segment_ms", 90.0), "min_segment_ms")
    min_gap_ms = _number(raw.get("min_gap_ms", 60.0), "min_gap_ms")
    if min_segment_ms < 0 or min_gap_ms < 0:
        raise BusinessError(E_CONFIG_INVALID, "minimum durations must be non-negative")
    quantile = str(raw.get("quantile_interpolation", "linear"))
    if quantile not in ALLOWED_QUANTILE_METHODS:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"unsupported quantile interpolation: {quantile}",
            allowed=sorted(ALLOWED_QUANTILE_METHODS),
        )

    dimensions_raw = raw.get("timbre_dimensions")
    if not isinstance(dimensions_raw, list) or not dimensions_raw:
        raise BusinessError(E_CONFIG_INVALID, "timbre_dimensions must be a non-empty array")
    dimensions: list[DimensionConfig] = []
    names: set[str] = set()
    for index, item in enumerate(dimensions_raw):
        if not isinstance(item, dict):
            raise BusinessError(E_CONFIG_INVALID, f"timbre_dimensions[{index}] must be an object")
        name = str(item.get("name", "")).strip()
        metric = str(item.get("metric", "")).strip()
        if not name or name in names:
            raise BusinessError(E_CONFIG_INVALID, f"timbre dimension name is empty or duplicated: {name!r}")
        if metric not in ALLOWED_METRICS:
            raise BusinessError(E_CONFIG_INVALID, f"unsupported metric {metric!r} for {name}")
        low, high = _pair(item.get("dynamic_range"), f"{name}.dynamic_range")
        if high <= low:
            raise BusinessError(E_CONFIG_INVALID, f"{name}.dynamic_range must increase")
        target_step = _number(item.get("target_step"), f"{name}.target_step")
        if target_step <= 0:
            raise BusinessError(E_CONFIG_INVALID, f"{name}.target_step must be positive")
        widths = item.get("bit_widths")
        if not isinstance(widths, list) or not widths:
            raise BusinessError(E_CONFIG_INVALID, f"{name}.bit_widths must be a non-empty array")
        try:
            bit_widths = tuple(sorted({int(value) for value in widths}))
        except (TypeError, ValueError, OverflowError) as exc:
            raise BusinessError(E_CONFIG_INVALID, f"{name}.bit_widths must contain integers") from exc
        if bit_widths[0] < 1 or bit_widths[-1] > 16:
            raise BusinessError(E_CONFIG_INVALID, f"{name}.bit_widths must be within 1..16")
        dimensions.append(DimensionConfig(name, metric, (low, high), target_step, bit_widths))
        names.add(name)

    similarity_raw = raw.get("similarity", {})
    threshold = _number(similarity_raw.get("threshold", raw.get("similarity_threshold", 0.86)), "similarity.threshold")
    if not 0.0 <= threshold <= 1.0:
        raise BusinessError(E_CONFIG_INVALID, "similarity threshold must be within [0, 1]")
    weights_raw = similarity_raw.get("weights", raw.get("similarity_weights", {})) or {}
    if not isinstance(weights_raw, dict):
        raise BusinessError(E_CONFIG_INVALID, "similarity.weights must be an object")
    weights: dict[str, float] = {}
    for name in [dimension.name for dimension in dimensions]:
        weight = _number(weights_raw.get(name, 1.0), f"similarity.weights.{name}")
        if weight < 0:
            raise BusinessError(E_CONFIG_INVALID, f"similarity weight for {name} must be non-negative")
        weights[name] = weight
    if sum(weights.values()) <= 0:
        raise BusinessError(E_CONFIG_INVALID, "at least one similarity weight must be positive")

    f0 = raw.get("f0", {})
    f0_min = _number(f0.get("min_hz", 60.0), "f0.min_hz")
    f0_max = _number(f0.get("max_hz", 500.0), "f0.max_hz")
    if not 0 < f0_min < f0_max < sample_rate / 2:
        raise BusinessError(E_CONFIG_INVALID, "f0 range must be positive and below Nyquist")
    formants = raw.get("formants", {})
    style = raw.get("style", {})
    rate_bins = tuple(_pair(style.get("speech_rate_bins_sps", [3.8, 5.0]), "style.speech_rate_bins_sps"))
    labels = style.get("labels", ["slow", "neutral", "fast"])
    if not isinstance(labels, list) or len(labels) != 3:
        raise BusinessError(E_CONFIG_INVALID, "style.labels must contain three labels")

    annotations = raw.get("initial_annotations", []) or []
    if not isinstance(annotations, list) or any(not isinstance(item, dict) for item in annotations):
        raise BusinessError(E_CONFIG_INVALID, "initial_annotations must be an array")

    lpc_order = _integer(formants.get("lpc_order", 16), "formants.lpc_order")
    if not 4 <= lpc_order <= 64:
        raise BusinessError(E_CONFIG_INVALID, "formants.lpc_order must be within 4..64")
    f1_range = _pair(formants.get("f1_range_hz", [180, 1100]), "formants.f1_range_hz")
    f2_range = _pair(formants.get("f2_range_hz", [700, 3500]), "formants.f2_range_hz")
    for field_name, frequency_range in (("f1_range_hz", f1_range), ("f2_range_hz", f2_range)):
        if not (0 < frequency_range[0] < frequency_range[1] < sample_rate / 2):
            raise BusinessError(E_CONFIG_INVALID, f"formants.{field_name} must be ordered and below Nyquist")
    if rate_bins[0] >= rate_bins[1]:
        raise BusinessError(E_CONFIG_INVALID, "style speech-rate bins must increase")
    high_variation = _number(style.get("high_variation_f0_std_hz", 45.0), "style.high_variation_f0_std_hz")
    if high_variation < 0:
        raise BusinessError(E_CONFIG_INVALID, "style.high_variation_f0_std_hz must be non-negative")
    max_workers = _integer(raw.get("max_workers", 4), "max_workers")
    if max_workers < 1:
        raise BusinessError(E_CONFIG_INVALID, "max_workers must be at least 1")
    commit_every = _integer(raw.get("snapshot_commit_every", 1), "snapshot_commit_every")
    if commit_every < 1:
        raise BusinessError(E_CONFIG_INVALID, "snapshot_commit_every must be at least 1")

    return AnalysisConfig(
        raw=copy.deepcopy(raw),
        source_path=source_path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width_bits=width,
        frame_length_ms=frame_length_ms,
        frame_shift_ms=frame_shift_ms,
        energy_enter_db=enter_db,
        energy_exit_db=exit_db,
        min_segment_ms=min_segment_ms,
        min_gap_ms=min_gap_ms,
        quantile_interpolation=quantile,
        f0_min_hz=f0_min,
        f0_max_hz=f0_max,
        voiced_autocorr=_number(f0.get("voiced_autocorr", 0.28), "f0.voiced_autocorr"),
        lpc_order=lpc_order,
        f1_range_hz=f1_range,
        f2_range_hz=f2_range,
        style_rate_bins_sps=(rate_bins[0], rate_bins[1]),
        style_labels=tuple(str(value) for value in labels),  # type: ignore[arg-type]
        high_variation_f0_std_hz=high_variation,
        high_variation_label=str(style.get("high_variation_label", "expressive")),
        dimensions=tuple(dimensions),
        similarity_threshold=threshold,
        similarity_weights=weights,
        preview_budget_ms=_number(raw.get("preview_budget_ms", 250.0), "preview_budget_ms"),
        max_workers=max_workers,
        snapshot_commit_every=commit_every,
        verify_incremental_vs_full=bool(raw.get("verify_incremental_vs_full", True)),
        initial_annotations=tuple(dict(item) for item in annotations),
        schema_version=schema_version,
    )


def load_config(path: str | Path) -> AnalysisConfig:
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BusinessError(E_CONFIG_INVALID, f"cannot read configuration: {exc}") from exc
    return parse_config(raw, config_path)
