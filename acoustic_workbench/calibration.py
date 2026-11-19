"""Frequency-domain calibration FIR design and simulated re-measurement."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy import signal

from .analysis import DelayAnalysis, MismatchAnalysis, analyze_mismatch, estimate_delays
from .audio import encode_wav_24
from .deconvolution import estimate_ir_set
from .dsp import smooth_band_mask
from .errors import ErrorCode
from .hashing import bytes_hash, content_hash
from .models import DeconvolutionResult, MeasurementRun, WorkbenchConfig
from .simulation import measurement_layout, propagate_channel


@dataclass(frozen=True)
class CalibrationFilter:
    channel_index: int
    channel: str
    coefficients: tuple[float, ...]
    delay_samples: int
    response_magnitude_db: tuple[float, ...]
    response_phase_deg: tuple[float, ...]
    maximum_abs_correction_db: float
    design_status: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationDesign:
    filters: tuple[CalibrationFilter, ...]
    filter_length: int
    frequencies_hz: tuple[float, ...]
    result_hash: str
    window: str


@dataclass(frozen=True)
class ResidualChannelResult:
    channel_index: int
    channel: str
    residual_gain_db: float
    residual_amplitude_db: tuple[float, ...]
    residual_phase_deg: tuple[float, ...]
    allowed_amplitude_halfwidth_db: tuple[float, ...]
    allowed_phase_halfwidth_deg: tuple[float, ...]
    max_amplitude_excess_db: float
    max_phase_excess_deg: float
    compatible: bool
    violation_bands: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CalibrationVerification:
    residual: MismatchAnalysis
    channels: tuple[ResidualChannelResult, ...]
    status: str
    errors: tuple[Mapping[str, Any], ...]
    remeasured_hash: str
    result_hash: str


def _window_values(name: str, length: int) -> np.ndarray:
    if name == "hann":
        return signal.windows.hann(length, sym=True)
    if name == "hamming":
        return signal.windows.hamming(length, sym=True)
    if name == "blackmanharris":
        return signal.windows.blackmanharris(length, sym=True)
    raise ValueError(f"unsupported calibration window: {name}")


def _clamp_correction(correction: np.ndarray, max_boost_db: float) -> np.ndarray:
    limit = 10.0 ** (max_boost_db / 20.0)
    values = np.nan_to_num(correction, nan=1.0, posinf=limit, neginf=-limit)
    magnitude = np.abs(values)
    unit_phase = values / np.maximum(magnitude, 1.0e-300)
    unit_phase[magnitude < 1.0e-280] = 1.0
    limited_magnitude = np.clip(magnitude, 1.0 / limit, limit)
    return (limited_magnitude * unit_phase).astype(np.complex128)


def design_calibration(
    run: MeasurementRun,
    deconvolution: DeconvolutionResult,
    mismatch: MismatchAnalysis,
    config: WorkbenchConfig,
) -> CalibrationDesign:
    calib = config.calibration
    reference_index = int(config.analysis.reference_channel)
    reference = deconvolution.channels[reference_index]
    frequencies = reference.frequencies_hz
    nfft = int(reference.nfft)
    length = int(calib.filter_length)
    delay = (length - 1) // 2
    if length > nfft // 2:
        from .errors import WorkbenchError

        raise WorkbenchError(
            ErrorCode.FILTER_LEN,
            "calibration filter is too long for the IR FFT grid",
            stage="calibration",
            details={"filter_length": length, "nfft": nfft},
        )
    window = _window_values(calib.window, length)
    band_mask = smooth_band_mask(
        frequencies,
        config.analysis.band_low_hz,
        config.analysis.band_high_hz,
        max(20.0, calib.edge_taper_fraction * (config.analysis.band_high_hz - config.analysis.band_low_hz)),
    )
    targets = np.asarray(mismatch.frequencies_hz, dtype=np.float64)
    target_indices = np.searchsorted(frequencies, targets)
    target_indices = np.clip(target_indices, 0, frequencies.size - 1)
    filters: list[CalibrationFilter] = []
    reference_spectrum = reference.transfer
    for channel_index, channel in enumerate(deconvolution.channels):
        if channel_index == reference_index:
            coefficients = np.zeros(length, dtype=np.float64)
            coefficients[delay] = 1.0
            effective = np.ones(frequencies.size, dtype=np.complex128)
            max_correction = 0.0
            iterations_used = 0
        else:
            denominator = np.abs(channel.transfer) ** 2 + calib.regularization * max(
                float(np.max(np.abs(reference_spectrum) ** 2)), 1.0e-30
            )
            raw_correction = reference_spectrum * np.conj(channel.transfer) / np.maximum(
                denominator, 1.0e-300
            )
            raw_correction = _clamp_correction(raw_correction, calib.max_boost_db)
            desired = 1.0 + band_mask * (raw_correction - 1.0)
            desired = _clamp_correction(desired, calib.max_boost_db)
            coefficients = np.zeros(length, dtype=np.float64)
            effective = np.ones(frequencies.size, dtype=np.complex128)
            for iteration in range(calib.design_iterations):
                linear_phase = np.exp(-2.0j * math.pi * frequencies * delay / run.config.sample_rate)
                spectrum = desired * linear_phase
                full_impulse = np.fft.irfft(spectrum, n=nfft)
                coefficients = np.asarray(full_impulse[:length] * window, dtype=np.float64)
                common_delay = np.exp(2.0j * math.pi * frequencies * delay / run.config.sample_rate)
                effective = np.fft.rfft(coefficients, n=nfft) * common_delay
                if iteration == calib.design_iterations - 1:
                    break
                safe_effective = np.where(np.abs(effective) < 1.0e-12, 1.0e-12, effective)
                residual_ratio = raw_correction / safe_effective
                update = 1.0 + band_mask * (residual_ratio - 1.0)
                update = _clamp_correction(update, calib.max_boost_db)
                adjusted = desired * (0.5 + 0.5 * update)
                desired = _clamp_correction(adjusted, calib.max_boost_db)
            iterations_used = calib.design_iterations
            correction_db = 20.0 * np.log10(np.maximum(np.abs(effective[band_mask > 0.5]), 1.0e-30))
            max_correction = float(np.max(np.abs(correction_db))) if correction_db.size else 0.0

        at_targets = np.interp(targets, frequencies, effective.real) + 1.0j * np.interp(
            targets, frequencies, effective.imag
        )
        response_mag = tuple(float(value) for value in 20.0 * np.log10(np.maximum(np.abs(at_targets), 1.0e-300)))
        response_phase = tuple(float(value) for value in np.degrees(np.angle(at_targets)))
        filters.append(
            CalibrationFilter(
                channel_index=channel_index,
                channel=run.config.channels[channel_index].name,
                coefficients=tuple(float(value) for value in coefficients),
                delay_samples=int(delay),
                response_magnitude_db=response_mag,
                response_phase_deg=response_phase,
                maximum_abs_correction_db=float(max_correction),
                design_status="OK",
                details={
                    "window": calib.window,
                    "regularization": calib.regularization,
                    "max_boost_db": calib.max_boost_db,
                    "design_iterations": iterations_used,
                    "application": "causal FIR with one common linear-phase delay across channels",
                },
            )
        )
    result_hash = content_hash(
        {
            "filter_length": length,
            "window": calib.window,
            "filters": [
                {
                    "channel": item.channel,
                    "coefficients": item.coefficients,
                    "delay": item.delay_samples,
                    "response_hz": targets,
                    "response_mag_db": item.response_magnitude_db,
                    "response_phase_deg": item.response_phase_deg,
                }
                for item in filters
            ],
            "deconvolution_hash": deconvolution.result_hash,
            "mismatch_hash": mismatch.result_hash,
        }
    )
    return CalibrationDesign(
        filters=tuple(filters),
        filter_length=length,
        frequencies_hz=tuple(float(value) for value in targets),
        result_hash=result_hash,
        window=calib.window,
    )


def simulate_calibrated_measurement(
    run: MeasurementRun,
    design: CalibrationDesign,
) -> MeasurementRun:
    """Apply each designed FIR to the drive and re-run the same room path."""

    config = run.config
    layout = measurement_layout(config)
    recorded = np.zeros((layout.n_frames, len(config.channels)), dtype=np.float64)
    delay = (design.filter_length - 1) // 2
    for channel_index, channel in enumerate(config.channels):
        coefficients = np.asarray(design.filters[channel_index].coefficients, dtype=np.float64)
        convolved = signal.fftconvolve(run.source, coefficients, mode="full")
        drive = np.asarray(convolved[: layout.n_frames], dtype=np.float64)
        recorded[:, channel_index], _ = propagate_channel(drive, channel, config)
    if float(np.max(np.abs(recorded))) >= 1.0:
        from .errors import WorkbenchError

        raise WorkbenchError(
            ErrorCode.AUDIO_FORMAT,
            "calibrated re-measurement would clip 24-bit PCM",
            stage="calibration",
        )
    wav = encode_wav_24(recorded, sample_rate=config.sample_rate, channels=len(config.channels), bit_depth=24)
    from .audio import decode_wav_24

    recorded, _ = decode_wav_24(wav)
    return MeasurementRun(
        config=config,
        source=run.source,
        recorded=recorded,
        wav_bytes=wav,
        sync_markers=run.sync_markers,
        sweep_starts=run.sweep_starts,
        source_hash=run.source_hash,
        recorded_hash=content_hash({"recorded": recorded}),
        wav_hash=bytes_hash(wav),
        noise_sigma_estimate=run.noise_sigma_estimate,
    )


def verify_calibration(
    run: MeasurementRun,
    calibrated_run: MeasurementRun,
    deconvolution: DeconvolutionResult,
    mismatch: MismatchAnalysis,
    delays: DelayAnalysis,
    design: CalibrationDesign,
    config: WorkbenchConfig,
    *,
    raise_on_mismatch: bool = False,
) -> CalibrationVerification:
    calibrated_deconv = estimate_ir_set(calibrated_run, config)
    calibrated_delays = estimate_delays(calibrated_run, calibrated_deconv, config, raise_on_bias=False)
    residual = analyze_mismatch(calibrated_run, calibrated_deconv, calibrated_delays, config)
    tolerance_factor = float(config.calibration.residual_tolerance_factor)
    channels: list[ResidualChannelResult] = []
    errors: list[Mapping[str, Any]] = []
    for channel_index, (residual_channel, initial_channel) in enumerate(zip(residual.channels, mismatch.channels)):
        amp_residual = np.asarray(residual_channel.amplitude_ratio_db, dtype=np.float64)
        phase_residual = np.asarray(residual_channel.phase_difference_deg, dtype=np.float64)
        allowed_amp = tolerance_factor * np.asarray(initial_channel.amplitude_error_halfwidth_db, dtype=np.float64)
        allowed_phase = tolerance_factor * np.asarray(initial_channel.phase_error_halfwidth_deg, dtype=np.float64)
        amp_excess = np.abs(amp_residual) - allowed_amp
        phase_excess = np.abs(phase_residual) - allowed_phase
        point_bands: list[dict[str, Any]] = []
        points = np.asarray(residual.frequencies_hz, dtype=np.float64)
        for point_index in range(points.size):
            if amp_excess[point_index] <= 0.0 and phase_excess[point_index] <= 0.0:
                continue
            width = float(points[1] - points[0]) if points.size > 1 else 1.0
            point_bands.append(
                {
                    "low_hz": float(max(residual.band_low_hz, points[point_index] - width / 2.0)),
                    "high_hz": float(min(residual.band_high_hz, points[point_index] + width / 2.0)),
                    "amplitude_excess_db": float(max(amp_excess[point_index], 0.0)),
                    "phase_excess_deg": float(max(phase_excess[point_index], 0.0)),
                }
            )
        violation_bands: list[Mapping[str, Any]] = []
        for item in point_bands:
            if not violation_bands:
                violation_bands.append(dict(item))
                continue
            previous = dict(violation_bands[-1])
            if abs(float(item["low_hz"]) - float(previous["high_hz"])) <= 1.0e-9:
                previous["high_hz"] = item["high_hz"]
                previous["amplitude_excess_db"] = max(
                    float(previous["amplitude_excess_db"]), float(item["amplitude_excess_db"])
                )
                previous["phase_excess_deg"] = max(
                    float(previous["phase_excess_deg"]), float(item["phase_excess_deg"])
                )
                violation_bands[-1] = previous
            else:
                violation_bands.append(dict(item))
        compatible = not violation_bands
        max_amp_excess = float(np.max(amp_excess)) if amp_excess.size else 0.0
        max_phase_excess = float(np.max(phase_excess)) if phase_excess.size else 0.0
        channels.append(
            ResidualChannelResult(
                channel_index=channel_index,
                channel=residual_channel.channel,
                residual_gain_db=float(residual_channel.weighted_gain_db),
                residual_amplitude_db=tuple(float(value) for value in amp_residual),
                residual_phase_deg=tuple(float(value) for value in phase_residual),
                allowed_amplitude_halfwidth_db=tuple(float(value) for value in allowed_amp),
                allowed_phase_halfwidth_deg=tuple(float(value) for value in allowed_phase),
                max_amplitude_excess_db=max_amp_excess,
                max_phase_excess_deg=max_phase_excess,
                compatible=compatible,
                violation_bands=tuple(violation_bands),
            )
        )
        if not compatible:
            errors.append(
                {
                    "code": ErrorCode.CALIB_MISMATCH.value,
                    "stage": "calibration-verification",
                    "message": "residual mismatch exceeds the pre-calibration uncertainty interval",
                    "details": {
                        "channel": residual_channel.channel,
                        "max_amplitude_excess_db": max_amp_excess,
                        "max_phase_excess_deg": max_phase_excess,
                        "bands": [dict(item) for item in violation_bands],
                    },
                }
            )
    status = "OK" if not errors else "FAIL"
    result_hash = content_hash(
        {
            "calibration_design_hash": design.result_hash,
            "remeasured_hash": calibrated_run.recorded_hash,
            "residual_hash": residual.result_hash,
            "channels": [
                {
                    "channel": item.channel,
                    "amplitude": item.residual_amplitude_db,
                    "phase": item.residual_phase_deg,
                    "compatible": item.compatible,
                }
                for item in channels
            ],
            "status": status,
        }
    )
    verification = CalibrationVerification(
        residual=residual,
        channels=tuple(channels),
        status=status,
        errors=tuple(errors),
        remeasured_hash=calibrated_run.recorded_hash,
        result_hash=result_hash,
    )
    if errors and raise_on_mismatch:
        from .errors import WorkbenchError

        raise WorkbenchError(
            ErrorCode.CALIB_MISMATCH,
            str(errors[0]["message"]),
            stage="calibration-verification",
            details=dict(errors[0]["details"]),
        )
    return verification
