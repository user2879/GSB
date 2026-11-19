"""Delay, gain/phase mismatch and propagated-uncertainty analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy import signal
from scipy import fft as sp_fft

from .dsp import quartic_peak_offset, smooth_band_mask, weighted_mean
from .errors import ErrorCode, WorkbenchError
from .hashing import content_hash
from .models import DeconvolutionResult, IRChannelResult, MeasurementRun, WorkbenchConfig


@dataclass(frozen=True)
class DelayChannelResult:
    channel_index: int
    channel: str
    truth_relative_samples: float
    estimated_relative_samples: float
    bias_samples: float
    error_bound_samples: float
    snr_db: float
    crb_samples: float
    curvature_relative: float
    interpolation_model_error_samples: float
    correlation_band_hz: tuple[float, float]
    status: str
    error_code: str | None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DelayAnalysis:
    reference_channel: int
    channels: tuple[DelayChannelResult, ...]
    errors: tuple[Mapping[str, Any], ...]
    result_hash: str
    formula: str


@dataclass(frozen=True)
class SubbandMismatch:
    index: int
    low_hz: float
    high_hz: float
    weight: float
    estimated_gain_db: float
    truth_gain_db: float
    bias_db: float
    noise_contribution_db: float
    leakage_contribution_db: float
    band_limit_contribution_db: float
    interval_low_db: float
    interval_high_db: float
    phase_interval_halfwidth_deg: float


@dataclass(frozen=True)
class ChannelMismatchResult:
    channel_index: int
    channel: str
    amplitude_ratio_db: tuple[float, ...]
    phase_difference_deg: tuple[float, ...]
    amplitude_error_halfwidth_db: tuple[float, ...]
    phase_error_halfwidth_deg: tuple[float, ...]
    noise_contribution_db: tuple[float, ...]
    leakage_contribution_db: tuple[float, ...]
    band_limit_contribution_db: tuple[float, ...]
    weighted_gain_db: float
    truth_gain_db: float
    gain_bias_db: float
    interval_low_db: float
    interval_high_db: float
    noise_bound_db: float
    leakage_bound_db: float
    band_limit_bound_db: float
    phase_interval_halfwidth_deg: float
    subbands: tuple[SubbandMismatch, ...]
    truth_within_interval: bool
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MismatchAnalysis:
    reference_channel: int
    frequencies_hz: tuple[float, ...]
    channels: tuple[ChannelMismatchResult, ...]
    band_low_hz: float
    band_high_hz: float
    point_weights: tuple[float, ...]
    result_hash: str
    error_model: Mapping[str, Any]


def _interp_complex(frequencies: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    real = np.interp(targets, frequencies, values.real)
    imaginary = np.interp(targets, frequencies, values.imag)
    return real + 1.0j * imaginary


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _find_window_peak(
    channel_ir: np.ndarray,
    reference_ir: np.ndarray,
    *,
    sample_rate: int,
    band_low_hz: float,
    band_high_hz: float,
    max_lag_samples: int = 96,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    nfft = int(sp_fft.next_fast_len(max(4096, 2 * max(channel_ir.size, reference_ir.size)), real=True))
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    taper = min(500.0, max(20.0, 0.12 * (band_high_hz - band_low_hz)))
    weights = smooth_band_mask(frequencies, band_low_hz, band_high_hz, taper)
    weights *= np.square(frequencies / max(band_high_hz, 1.0))
    spectrum_channel = np.fft.rfft(channel_ir, n=nfft)
    spectrum_reference = np.fft.rfft(reference_ir, n=nfft)
    correlation = np.fft.irfft(spectrum_channel * np.conj(spectrum_reference) * weights, n=nfft)
    envelope = np.abs(signal.hilbert(correlation))
    shifted = np.concatenate((envelope[-max_lag_samples:], envelope[: max_lag_samples + 1]))
    lags = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=np.float64)
    peak = int(np.argmax(shifted))
    curvature = 0.0
    delta = 0.0
    if 0 < peak < shifted.size - 1:
        left, center, right = shifted[peak - 1 : peak + 2]
        curvature = float(left - 2.0 * center + right)
        if abs(curvature) > 1.0e-30:
            delta = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return float(lags[peak] + delta), lags, shifted, correlation, curvature


def estimate_delays(
    run: MeasurementRun,
    deconvolution: DeconvolutionResult,
    config: WorkbenchConfig,
    *,
    raise_on_bias: bool = True,
) -> DelayAnalysis:
    analysis = config.analysis
    reference_index = int(analysis.reference_channel)
    reference = deconvolution.channels[reference_index]
    low_hz = max(200.0, config.deconvolution.effective_low_hz)
    high_hz = min(16_000.0, config.deconvolution.effective_high_hz)
    if high_hz <= low_hz:
        low_hz = max(20.0, config.deconvolution.effective_low_hz)
        high_hz = min(18_000.0, config.deconvolution.effective_high_hz)

    source_segment = run.source[run.sweep_starts[0] : run.sweep_starts[0] + int(round(config.measurement.sweep_duration_s * run.config.sample_rate))]
    gradient_ratio = float(np.sum(np.diff(source_segment) ** 2) / max(np.sum(source_segment**2), 1.0e-30))

    channel_results: list[DelayChannelResult] = []
    errors: list[Mapping[str, Any]] = []
    for channel_index, channel in enumerate(deconvolution.channels):
        truth = float(run.config.channels[channel_index].delay_samples - run.config.channels[reference_index].delay_samples)
        if channel_index == reference_index:
            estimated = 0.0
            curvature = 0.0
            model_error = 0.0
            snr_db = float(channel.snr_db)
        else:
            estimated, lags, envelope, _, curvature = _find_window_peak(
                channel.ir,
                reference.ir,
                sample_rate=run.config.sample_rate,
                band_low_hz=low_hz,
                band_high_hz=high_hz,
            )
            peak = int(np.argmax(envelope))
            quartic = quartic_peak_offset(envelope, peak)
            three_point_offset = float(estimated - lags[peak])
            model_error = abs(quartic - three_point_offset)
            noise_sigma = float(run.noise_sigma_estimate[channel_index])
            signal_rms = float(channel.signal_rms)
            snr_linear = max((signal_rms / max(noise_sigma, 1.0e-30)) ** 2, 1.0e-30)
            snr_db = 10.0 * math.log10(snr_linear)

        noise_sigma = float(run.noise_sigma_estimate[channel_index])
        snr_linear = max((float(channel.signal_rms) / max(noise_sigma, 1.0e-30)) ** 2, 1.0e-30)
        crb = math.sqrt(1.0 / max(snr_linear * gradient_ratio, 1.0e-30))
        peak_scale = float(envelope[np.argmax(envelope)]) if channel_index != reference_index else 1.0
        curvature_relative = abs(curvature / max(peak_scale, 1.0e-30)) if channel_index != reference_index else 0.0
        curvature_bias = 0.125 * curvature_relative
        bound = analysis.coverage_sigma * math.sqrt(crb * crb + curvature_bias * curvature_bias) + 2.0 * model_error
        quantization_bound = (2.0 ** -23) / max(float(channel.signal_rms), 1.0e-30)
        bound = float(bound + quantization_bound)
        bias = float(estimated - truth)
        status = "OK"
        error_code: str | None = None
        if abs(truth) > analysis.delay_tolerance_samples:
            status = "FAIL"
            error_code = ErrorCode.DELAY_BIAS.value
        elif abs(bias) > max(analysis.delay_bias_tolerance_samples, bound):
            status = "FAIL"
            error_code = ErrorCode.DELAY_BIAS.value
        details = {
            "reference_channel": reference_index,
            "configured_delay_samples": float(run.config.channels[channel_index].delay_samples),
            "reference_delay_samples": float(run.config.channels[reference_index].delay_samples),
            "delay_tolerance_samples": analysis.delay_tolerance_samples,
            "bias_tolerance_samples": analysis.delay_bias_tolerance_samples,
            "within_error_bound": abs(bias) <= bound,
            "within_tolerance": abs(bias) <= analysis.delay_bias_tolerance_samples,
        }
        result = DelayChannelResult(
            channel_index=channel_index,
            channel=run.config.channels[channel_index].name,
            truth_relative_samples=truth,
            estimated_relative_samples=float(estimated),
            bias_samples=bias,
            error_bound_samples=bound,
            snr_db=float(snr_db),
            crb_samples=float(crb),
            curvature_relative=float(curvature_relative),
            interpolation_model_error_samples=float(model_error),
            correlation_band_hz=(float(low_hz), float(high_hz)),
            status=status,
            error_code=error_code,
            details=details,
        )
        channel_results.append(result)
        if status != "OK":
            event = {
                "code": error_code,
                "stage": "delay",
                "message": "configured delay or truth/estimate bias exceeds tolerance",
                "details": {
                    "channel": result.channel,
                    "truth_relative_samples": truth,
                    "estimated_relative_samples": estimated,
                    "bias_samples": bias,
                    "error_bound_samples": bound,
                    "delay_tolerance_samples": analysis.delay_tolerance_samples,
                },
            }
            errors.append(event)

    result_hash = content_hash(
        {
            "reference_channel": reference_index,
            "channels": [result.details | {
                "estimated": result.estimated_relative_samples,
                "truth": result.truth_relative_samples,
                "bound": result.error_bound_samples,
                "status": result.status,
            } for result in channel_results],
            "deconvolution_hash": deconvolution.result_hash,
        }
    )
    analysis_result = DelayAnalysis(
        reference_channel=reference_index,
        channels=tuple(channel_results),
        errors=tuple(errors),
        result_hash=result_hash,
        formula=(
            "bound = coverage*sqrt(CRB^2 + (0.125*curvature_relative)^2) + 2*quartic_model_error + quantization"
        ),
    )
    if errors and raise_on_bias:
        first = errors[0]
        raise WorkbenchError(
            ErrorCode.DELAY_BIAS,
            str(first["message"]),
            stage="delay",
            details=dict(first["details"]),
        )
    return analysis_result


def _weights_for_gain(reference_spectrum: np.ndarray) -> np.ndarray:
    weights = np.abs(reference_spectrum) ** 2
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(weights.shape, 1.0 / weights.size, dtype=np.float64)
    return weights / total


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    normalized = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(normalized))
    if total <= 0.0:
        return float(np.sqrt(np.mean(np.square(values))))
    normalized = normalized / total
    return float(np.sqrt(np.sum(normalized * np.square(values))))


def analyze_mismatch(
    run: MeasurementRun,
    deconvolution: DeconvolutionResult,
    delays: DelayAnalysis,
    config: WorkbenchConfig,
) -> MismatchAnalysis:
    analysis = config.analysis
    reference_index = int(analysis.reference_channel)
    reference = deconvolution.channels[reference_index]
    targets = np.linspace(analysis.band_low_hz, analysis.band_high_hz, analysis.points, dtype=np.float64)
    reference_spectrum = _interp_complex(reference.frequencies_hz, reference.transfer, targets)
    weights = _weights_for_gain(reference_spectrum)
    effective_span = max(config.deconvolution.effective_high_hz - config.deconvolution.effective_low_hz, 1.0)
    analysis_span = max(analysis.band_high_hz - analysis.band_low_hz, 1.0)
    span_ratio = float(np.clip(analysis_span / effective_span, 1.0e-6, 1.0))
    delay_bounds = {result.channel_index: result.error_bound_samples for result in delays.channels}
    sample_rate = float(run.config.sample_rate)

    channel_results: list[ChannelMismatchResult] = []
    for channel_index, channel in enumerate(deconvolution.channels):
        spectrum = _interp_complex(channel.frequencies_hz, channel.transfer, targets)
        amplitude_db = 20.0 * np.log10(np.maximum(np.abs(spectrum), 1.0e-300) / np.maximum(np.abs(reference_spectrum), 1.0e-300))
        phase_deg = _wrap_degrees(np.degrees(np.angle(spectrum * np.conj(reference_spectrum))))
        weighted_gain = weighted_mean(amplitude_db, weights)

        relative_noise_pointwise = np.sqrt(
            np.interp(targets, channel.frequencies_hz, channel.transfer_relative_noise) ** 2
            + np.interp(targets, reference.frequencies_hz, reference.transfer_relative_noise) ** 2
        )
        relative_noise = float(np.sqrt(np.mean(np.square(relative_noise_pointwise))))
        noise_contrib_db = (20.0 / math.log(10.0)) * relative_noise_pointwise

        truncation_leakage_db = 10.0 * math.log10(max(1.0 + channel.truncation_energy_fraction, 1.0))
        leakage_level_db = max(0.0, channel.window_leakage_db + truncation_leakage_db)
        leakage_contrib_db = np.full(targets.size, leakage_level_db, dtype=np.float64)
        leakage_phase_deg = leakage_level_db * (360.0 * targets / sample_rate)

        log_frequency = np.log2(np.maximum(targets, 1.0))
        smooth_fit = np.polyval(np.polyfit(log_frequency, amplitude_db, 2), log_frequency)
        frequency_shape = np.abs(amplitude_db - smooth_fit)
        band_contrib_db = (
            0.20 * frequency_shape
            + 0.005 * math.sqrt(span_ratio)
            + 0.01 * math.sqrt(span_ratio) * np.abs(amplitude_db - weighted_gain) / max(targets.size, 1)
        )
        band_contrib_db = np.maximum(band_contrib_db, 1.0e-6)

        relative_delay_bound = math.sqrt(
            delay_bounds.get(channel_index, 0.0) ** 2 + delay_bounds.get(reference_index, 0.0) ** 2
        )
        delay_phase_sigma = 360.0 * targets * relative_delay_bound / sample_rate
        phase_noise_sigma = np.degrees(relative_noise_pointwise)
        phase_band_component = 0.03 * frequency_shape * (360.0 * targets / sample_rate) * math.sqrt(span_ratio)
        phase_halfwidth = analysis.coverage_sigma * np.sqrt(
            phase_noise_sigma**2 + delay_phase_sigma**2 + leakage_phase_deg**2 + phase_band_component**2
        )
        phase_halfwidth = np.maximum(phase_halfwidth, 1.0e-6)

        amplitude_halfwidth = analysis.coverage_sigma * np.sqrt(
            noise_contrib_db**2 + leakage_contrib_db**2 + band_contrib_db**2
        )
        noise_bound = analysis.coverage_sigma * _weighted_rms(noise_contrib_db, weights)
        leakage_bound = analysis.coverage_sigma * abs(weighted_mean(leakage_contrib_db, weights)) * math.sqrt(span_ratio)
        band_bound = analysis.coverage_sigma * abs(weighted_mean(band_contrib_db, weights)) * math.sqrt(span_ratio)
        interval_half = math.sqrt(noise_bound**2 + leakage_bound**2 + band_bound**2)
        interval_half = max(interval_half, 1.0e-6)
        phase_interval = analysis.coverage_sigma * _weighted_rms(phase_halfwidth, weights) * math.sqrt(span_ratio)

        subband_edges = np.geomspace(analysis.band_low_hz, analysis.band_high_hz, analysis.subband_count + 1)
        subbands: list[SubbandMismatch] = []
        for subband_index in range(analysis.subband_count):
            lower = float(subband_edges[subband_index])
            upper = float(subband_edges[subband_index + 1])
            if subband_index == analysis.subband_count - 1:
                mask = (targets >= lower) & (targets <= upper)
            else:
                mask = (targets >= lower) & (targets < upper)
            if not np.any(mask):
                continue
            sub_weights = weights[mask]
            if float(np.sum(sub_weights)) <= 0.0:
                sub_weights = np.full(sub_weights.shape, 1.0 / sub_weights.size, dtype=np.float64)
            sub_weight = float(np.sum(sub_weights))
            sub_gain = weighted_mean(amplitude_db[mask], sub_weights)
            truth_gain = 20.0 * math.log10(
                max(run.config.channels[channel_index].gain, 1.0e-30)
                / max(run.config.channels[reference_index].gain, 1.0e-30)
            )
            sub_noise = analysis.coverage_sigma * _weighted_rms(noise_contrib_db[mask], sub_weights)
            sub_span_ratio = float(np.clip((upper - lower) / effective_span, 1.0e-6, 1.0))
            sub_leak = analysis.coverage_sigma * abs(weighted_mean(leakage_contrib_db[mask], sub_weights)) * math.sqrt(sub_span_ratio)
            sub_band = analysis.coverage_sigma * abs(weighted_mean(band_contrib_db[mask], sub_weights)) * math.sqrt(sub_span_ratio)
            sub_half = math.sqrt(sub_noise**2 + sub_leak**2 + sub_band**2)
            sub_phase = analysis.coverage_sigma * _weighted_rms(phase_halfwidth[mask], sub_weights) * math.sqrt(sub_span_ratio)
            subbands.append(
                SubbandMismatch(
                    index=subband_index,
                    low_hz=lower,
                    high_hz=upper,
                    weight=sub_weight,
                    estimated_gain_db=sub_gain,
                    truth_gain_db=truth_gain,
                    bias_db=sub_gain - truth_gain,
                    noise_contribution_db=sub_noise,
                    leakage_contribution_db=sub_leak,
                    band_limit_contribution_db=sub_band,
                    interval_low_db=sub_gain - sub_half,
                    interval_high_db=sub_gain + sub_half,
                    phase_interval_halfwidth_deg=sub_phase,
                )
            )

        truth_gain = 20.0 * math.log10(
            max(run.config.channels[channel_index].gain, 1.0e-30)
            / max(run.config.channels[reference_index].gain, 1.0e-30)
        )
        channel_results.append(
            ChannelMismatchResult(
                channel_index=channel_index,
                channel=run.config.channels[channel_index].name,
                amplitude_ratio_db=tuple(float(value) for value in amplitude_db),
                phase_difference_deg=tuple(float(value) for value in phase_deg),
                amplitude_error_halfwidth_db=tuple(float(value) for value in amplitude_halfwidth),
                phase_error_halfwidth_deg=tuple(float(value) for value in phase_halfwidth),
                noise_contribution_db=tuple(float(value) for value in noise_contrib_db),
                leakage_contribution_db=tuple(float(value) for value in leakage_contrib_db),
                band_limit_contribution_db=tuple(float(value) for value in band_contrib_db),
                weighted_gain_db=float(weighted_gain),
                truth_gain_db=float(truth_gain),
                gain_bias_db=float(weighted_gain - truth_gain),
                interval_low_db=float(weighted_gain - interval_half),
                interval_high_db=float(weighted_gain + interval_half),
                noise_bound_db=float(noise_bound),
                leakage_bound_db=float(leakage_bound),
                band_limit_bound_db=float(band_bound),
                phase_interval_halfwidth_deg=float(phase_interval),
                subbands=tuple(subbands),
                truth_within_interval=bool(weighted_gain - interval_half <= truth_gain <= weighted_gain + interval_half),
                details={
                    "analysis_band_hz": [analysis.band_low_hz, analysis.band_high_hz],
                    "analysis_span_ratio": span_ratio,
                    "point_count": analysis.points,
                    "delay_bound_samples": relative_delay_bound,
                },
            )
        )

    result_hash = content_hash(
        {
            "frequencies": targets,
            "weights": weights,
            "channels": [
                {
                    "channel": channel.channel,
                    "amplitude": channel.amplitude_ratio_db,
                    "phase": channel.phase_difference_deg,
                    "interval": [channel.interval_low_db, channel.interval_high_db],
                    "weighted_gain": channel.weighted_gain_db,
                }
                for channel in channel_results
            ],
            "deconvolution_hash": deconvolution.result_hash,
            "delay_hash": delays.result_hash,
        }
    )
    return MismatchAnalysis(
        reference_channel=reference_index,
        frequencies_hz=tuple(float(value) for value in targets),
        channels=tuple(channel_results),
        band_low_hz=float(analysis.band_low_hz),
        band_high_hz=float(analysis.band_high_hz),
        point_weights=tuple(float(value) for value in weights),
        result_hash=result_hash,
        error_model={
            "formula": "sqrt(noise^2 + truncation_leakage^2 + band_limit^2), then weighted and span-normalized",
            "coverage_sigma": analysis.coverage_sigma,
            "span_ratio": span_ratio,
            "noise_source": "robust noise estimate from quiet pre-roll differentiated to suppress filter ringing",
            "leakage_source": "truncated-IR FFT versus full regularized transfer",
            "band_limit_source": "analysis-band frequency-shape and edge deviations",
        },
    )
