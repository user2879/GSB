"""Configuration and stage result models."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .errors import ErrorCode, fail


@dataclass(frozen=True)
class ChannelConfig:
    name: str
    delay_samples: float
    gain: float
    band_low_hz: float
    band_high_hz: float
    noise_seed: int
    noise_rms: float


@dataclass(frozen=True)
class MeasurementConfig:
    sample_rate: int = 48_000
    bit_depth: int = 24
    channels: tuple[ChannelConfig, ...] = field(default_factory=tuple)
    sweep_low_hz: float = 20.0
    sweep_high_hz: float = 20_000.0
    sweep_duration_s: float = 2.5
    repeats: int = 2
    lead_in_s: float = 0.05
    guard_pre_s: float = 0.02
    guard_post_s: float = 0.12
    fade_s: float = 0.015
    drive_amplitude: float = 0.20
    sync_pulse_amplitude: float = 0.36
    sync_pulse_center_hz: float = 4_800.0
    sync_pulse_sigma_samples: float = 4.0
    sync_pulse_radius_samples: int = 16
    master_seed: int = 20_261_119


@dataclass(frozen=True)
class DeconvolutionConfig:
    regularization: float = 1.0e-6
    effective_low_hz: float = 30.0
    effective_high_hz: float = 19_500.0
    nfft: int = 0
    window: str = "tukey"
    tukey_alpha: float = 0.02
    pre_margin_samples: int = 512
    post_margin_samples: int = 2_048
    truncation_samples: int = 4_096
    band_taper_hz: float = 120.0


@dataclass(frozen=True)
class AnalysisConfig:
    reference_channel: int = 0
    band_low_hz: float = 200.0
    band_high_hz: float = 12_000.0
    points: int = 96
    subband_count: int = 3
    coverage_sigma: float = 3.0
    delay_tolerance_samples: float = 1.0
    delay_bias_tolerance_samples: float = 0.08
    excitation_gap_guard_hz: float = 20.0


@dataclass(frozen=True)
class CalibrationConfig:
    filter_length: int = 1_025
    window: str = "blackmanharris"
    regularization: float = 2.0e-4
    max_boost_db: float = 12.0
    edge_taper_fraction: float = 0.08
    residual_tolerance_factor: float = 1.25
    design_iterations: int = 3


@dataclass(frozen=True)
class WorkbenchConfig:
    measurement: MeasurementConfig
    deconvolution: DeconvolutionConfig = field(default_factory=DeconvolutionConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    def with_measurement(self, **changes: Any) -> "WorkbenchConfig":
        return replace(self, measurement=replace(self.measurement, **changes))

    def with_deconvolution(self, **changes: Any) -> "WorkbenchConfig":
        return replace(self, deconvolution=replace(self.deconvolution, **changes))

    def with_analysis(self, **changes: Any) -> "WorkbenchConfig":
        return replace(self, analysis=replace(self.analysis, **changes))

    def with_calibration(self, **changes: Any) -> "WorkbenchConfig":
        return replace(self, calibration=replace(self.calibration, **changes))


def default_channels() -> tuple[ChannelConfig, ...]:
    return (
        ChannelConfig("CH0_REF", 0.0, 1.00, 25.0, 19_500.0, 90_001, 2.0e-4),
        ChannelConfig("CH1", 0.35, 0.82, 32.0, 18_200.0, 90_002, 2.0e-4),
        ChannelConfig("CH2", -0.22, 1.15, 45.0, 16_800.0, 90_003, 2.0e-4),
        ChannelConfig("CH3", 0.71, 0.91, 38.0, 17_600.0, 90_004, 2.0e-4),
    )


def default_config() -> WorkbenchConfig:
    return WorkbenchConfig(measurement=MeasurementConfig(channels=default_channels()))


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def validate_config(config: WorkbenchConfig) -> None:
    m = config.measurement
    d = config.deconvolution
    a = config.analysis
    c = config.calibration

    if m.sample_rate != 48_000 or m.bit_depth != 24 or len(m.channels) != 4:
        fail(
            ErrorCode.AUDIO_FORMAT,
            "measurement format must be 48 kHz, 24-bit, 4 channels",
            stage="validate",
            details={
                "sample_rate": m.sample_rate,
                "bit_depth": m.bit_depth,
                "channels": len(m.channels),
            },
        )
    if m.repeats < 1 or m.sweep_duration_s <= 0.0:
        fail(
            ErrorCode.BAND_INVALID,
            "repetitions and sweep duration must be positive",
            stage="validate",
        )
    nyquist = m.sample_rate / 2.0
    if not (0.0 < m.sweep_low_hz < m.sweep_high_hz < nyquist):
        fail(
            ErrorCode.BAND_INVALID,
            "sweep band must be inside (0, Nyquist)",
            stage="validate",
            details={"low_hz": m.sweep_low_hz, "high_hz": m.sweep_high_hz},
        )
    if not (0.0 < d.effective_low_hz < d.effective_high_hz <= nyquist):
        fail(
            ErrorCode.BAND_INVALID,
            "deconvolution band must be inside (0, Nyquist)",
            stage="validate",
            details={
                "low_hz": d.effective_low_hz,
                "high_hz": d.effective_high_hz,
            },
        )
    if not (0.0 < a.band_low_hz < a.band_high_hz <= nyquist):
        fail(
            ErrorCode.BAND_INVALID,
            "analysis band must be inside (0, Nyquist)",
            stage="validate",
            details={"low_hz": a.band_low_hz, "high_hz": a.band_high_hz},
        )
    if not (0 <= a.reference_channel < len(m.channels)):
        fail(
            ErrorCode.BAND_INVALID,
            "reference channel index is outside the channel list",
            stage="validate",
            details={"reference_channel": a.reference_channel},
        )
    if a.points < 8 or a.subband_count < 1 or a.subband_count > a.points:
        fail(ErrorCode.BAND_INVALID, "analysis point/subband count is invalid", stage="validate")
    if d.regularization <= 0.0 or d.truncation_samples < 64 or d.nfft < 0:
        fail(ErrorCode.BAND_INVALID, "deconvolution parameters are invalid", stage="validate")
    if c.filter_length < 33 or c.filter_length % 2 == 0 or c.filter_length > 8_193:
        fail(
            ErrorCode.FILTER_LEN,
            "calibration filter length must be odd in [33, 8193]",
            stage="validate",
            details={"filter_length": c.filter_length},
        )
    if c.design_iterations < 1 or c.design_iterations > 8:
        fail(ErrorCode.FILTER_LEN, "calibration design iterations must be in [1, 8]", stage="validate")
    for channel in m.channels:
        if not all(
            _is_finite_number(value)
            for value in (
                channel.delay_samples,
                channel.gain,
                channel.band_low_hz,
                channel.band_high_hz,
                channel.noise_rms,
            )
        ):
            fail(ErrorCode.AUDIO_FORMAT, "channel values must be finite", stage="validate")
        if channel.gain <= 0.0 or channel.noise_rms < 0.0:
            fail(ErrorCode.AUDIO_FORMAT, "channel gain/noise must be non-negative", stage="validate")
        if not (0.0 < channel.band_low_hz < channel.band_high_hz < nyquist):
            fail(
                ErrorCode.BAND_INVALID,
                f"channel band is invalid for {channel.name}",
                stage="validate",
                details={"channel": channel.name},
            )


def config_fingerprint_parts(config: WorkbenchConfig) -> dict[str, Any]:
    validate_config(config)
    return {
        "measurement": config.measurement,
        "deconvolution": config.deconvolution,
        "analysis": config.analysis,
        "calibration": config.calibration,
    }


@dataclass
class MeasurementRun:
    config: MeasurementConfig
    source: np.ndarray
    recorded: np.ndarray
    wav_bytes: bytes
    sync_markers: tuple[dict[str, Any], ...]
    sweep_starts: tuple[int, ...]
    source_hash: str
    recorded_hash: str
    wav_hash: str
    noise_sigma_estimate: np.ndarray


@dataclass
class IRChannelResult:
    channel_index: int
    ir: np.ndarray
    time_axis_samples: np.ndarray
    frequencies_hz: np.ndarray
    transfer: np.ndarray
    peak_index: int
    peak_sample: int
    arrival_sample: float
    peak_delta_samples: float
    peak_level_linear: float
    peak_level_db: float
    snr_db: float
    nfft: int
    window: str
    truncation_samples: int
    origin_shift_samples: int
    segment_offsets: tuple[int, ...]
    window_starts: tuple[int, ...]
    normalization_factor: float
    residual_rms: float
    signal_rms: float
    transfer_relative_noise: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    truncation_energy_fraction: float = 0.0
    window_leakage_db: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ir_hash(self) -> str:
        from .hashing import content_hash

        return content_hash(
            {
                "channel_index": self.channel_index,
                "ir": self.ir,
                "time_axis_samples": self.time_axis_samples,
                "transfer": self.transfer,
                "peak_sample": self.peak_sample,
                "arrival_sample": self.arrival_sample,
                "peak_level_db": self.peak_level_db,
                "nfft": self.nfft,
                "window": self.window,
                "truncation_samples": self.truncation_samples,
                "origin_shift_samples": self.origin_shift_samples,
                "window_starts": self.window_starts,
                "normalization_factor": self.normalization_factor,
                "residual_rms": self.residual_rms,
                "signal_rms": self.signal_rms,
                "transfer_relative_noise": self.transfer_relative_noise,
                "truncation_energy_fraction": self.truncation_energy_fraction,
                "window_leakage_db": self.window_leakage_db,
            }
        )


@dataclass
class DeconvolutionResult:
    config: DeconvolutionConfig
    channels: tuple[IRChannelResult, ...]
    nfft: int
    result_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


def as_serializable(value: Any) -> Any:
    """Convert a dataclass/array result to JSON-friendly values."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses_is_dataclass(value):
        return {field.name: as_serializable(getattr(value, field.name)) for field in value.__dataclass_fields__.values()}
    if isinstance(value, dict):
        return {str(key): as_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_serializable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def config_from_mapping(payload: Mapping[str, Any]) -> WorkbenchConfig:
    measurement_data = dict(payload.get("measurement", {}))
    channel_rows = measurement_data.pop("channels", [])
    channels = tuple(
        ChannelConfig(
            name=str(row["name"]),
            delay_samples=float(row["delay_samples"]),
            gain=float(row["gain"]),
            band_low_hz=float(row["band_low_hz"]),
            band_high_hz=float(row["band_high_hz"]),
            noise_seed=int(row["noise_seed"]),
            noise_rms=float(row["noise_rms"]),
        )
        for row in channel_rows
    )
    measurement = MeasurementConfig(channels=channels, **measurement_data)
    deconv_data = dict(payload.get("deconvolution", {}))
    analysis_data = dict(payload.get("analysis", {}))
    calibration_data = dict(payload.get("calibration", {}))
    return WorkbenchConfig(
        measurement=measurement,
        deconvolution=DeconvolutionConfig(**deconv_data),
        analysis=AnalysisConfig(**analysis_data),
        calibration=CalibrationConfig(**calibration_data),
    )


def dataclasses_is_dataclass(value: Any) -> bool:
    import dataclasses

    return dataclasses.is_dataclass(value)
