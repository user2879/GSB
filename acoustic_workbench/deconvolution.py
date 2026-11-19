"""Regularized frequency-domain sweep deconvolution."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import fft as sp_fft

from .dsp import smooth_band_mask, tukey_window
from .errors import ErrorCode, fail
from .hashing import content_hash
from .models import (
    DeconvolutionConfig,
    DeconvolutionResult,
    IRChannelResult,
    MeasurementRun,
    WorkbenchConfig,
)
from .simulation import measurement_layout


def _window_values(name: str, count: int, tukey_alpha: float) -> np.ndarray:
    if name == "rectangular":
        return np.ones(count, dtype=np.float64)
    if name == "tukey":
        return tukey_window(count, tukey_alpha)
    fail(
        ErrorCode.BAND_INVALID,
        f"unsupported deconvolution window: {name}",
        stage="deconvolution",
        details={"window": name},
    )
    raise AssertionError("unreachable")


def _selected_nfft(configured: int, minimum: int) -> int:
    if configured > 0:
        if configured < minimum:
            fail(
                ErrorCode.BAND_INVALID,
                "configured FFT length is shorter than the analysis window",
                stage="deconvolution",
                details={"nfft": configured, "minimum": minimum},
            )
        return int(configured)
    target = max(minimum, 4096)
    return int(sp_fft.next_fast_len(target, real=True))


def _raw_channel_estimate(
    run: MeasurementRun,
    channel_index: int,
    deconv_config: DeconvolutionConfig,
) -> dict[str, Any]:
    config = run.config
    layout = measurement_layout(config)
    pre = deconv_config.pre_margin_samples
    post = deconv_config.post_margin_samples
    win_len = layout.n_sweep + pre + post
    if pre <= 0 or post <= 0:
        fail(ErrorCode.BAND_INVALID, "deconvolution margins must be positive", stage="deconvolution")

    window_starts: list[int] = []
    for sweep_start in layout.sweep_starts:
        start = sweep_start - pre
        if start < 0 or start + win_len > run.recorded.shape[0]:
            fail(
                ErrorCode.BAND_INVALID,
                "deconvolution window exceeds the recorded signal",
                stage="deconvolution",
                details={"start": start, "length": win_len, "frames": run.recorded.shape[0]},
            )
        window_starts.append(int(start))

    nfft = _selected_nfft(deconv_config.nfft, win_len)
    if nfft % 2 != 0:
        nfft += 1
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / config.sample_rate)
    band_mask = smooth_band_mask(
        frequencies,
        deconv_config.effective_low_hz,
        deconv_config.effective_high_hz,
        deconv_config.band_taper_hz,
    )

    transfer_accumulator = np.zeros(frequencies.size, dtype=np.complex128)
    noise_amplification = np.zeros(frequencies.size, dtype=np.float64)
    signal_energy = 0.0
    residual_energy = 0.0
    sample_count = 0
    window_leakage_db = 0.0
    for start in window_starts:
        x = np.array(run.source[start : start + win_len], dtype=np.float64, copy=True)
        y = np.array(run.recorded[start : start + win_len, channel_index], dtype=np.float64, copy=True)
        taper = _window_values(deconv_config.window, win_len, deconv_config.tukey_alpha)
        window_leakage_db = max(
            window_leakage_db,
            float(10.0 * np.log10(max(1.0 + np.mean(np.square(1.0 - taper)) / max(np.mean(taper) ** 2, 1.0e-30), 1.0))),
        )
        x *= taper
        y *= taper
        spectrum_x = np.fft.rfft(x, n=nfft)
        spectrum_y = np.fft.rfft(y, n=nfft)
        power = np.abs(spectrum_x) ** 2
        regularization = float(deconv_config.regularization) * max(float(np.max(power)), 1.0e-30)
        denominator = power + regularization
        transfer = spectrum_y * np.conj(spectrum_x) / denominator
        transfer *= band_mask
        transfer_accumulator += transfer
        noise_amplification += power / np.square(denominator)
        predicted = np.fft.irfft(spectrum_x * transfer, n=nfft)
        residual = y - predicted[:win_len]
        sample_count += residual.size
        signal_energy += float(np.sum(y * y))
        residual_energy += float(np.sum(residual * residual))

    transfer = transfer_accumulator / float(len(window_starts))
    noise_sigma = float(run.noise_sigma_estimate[channel_index])
    transfer_noise_sigma = noise_sigma * np.sqrt(noise_amplification) / float(len(window_starts))
    transfer_relative_noise = transfer_noise_sigma / np.maximum(np.abs(transfer), 1.0e-30)
    transfer_relative_noise = np.minimum(transfer_relative_noise, 10.0)
    ir_raw = np.fft.irfft(transfer, n=nfft)
    origin_shift = int(deconv_config.pre_margin_samples)
    ir_rolled = np.roll(ir_raw, origin_shift)
    truncation = int(deconv_config.truncation_samples)
    if truncation > nfft:
        truncation = nfft
    ir = np.asarray(ir_rolled[:truncation], dtype=np.float64)
    time_axis = np.arange(truncation, dtype=np.int64) - origin_shift
    total_ir_energy = float(np.sum(np.square(ir_rolled)))
    kept_ir_energy = float(np.sum(np.square(ir)))
    truncation_energy_fraction = float(np.clip(1.0 - kept_ir_energy / max(total_ir_energy, 1.0e-30), 0.0, 1.0))

    signal_rms = math.sqrt(signal_energy / max(sample_count, 1))
    residual_rms = math.sqrt(residual_energy / max(sample_count, 1))
    snr_db = 20.0 * math.log10(max(signal_rms, 1.0e-300) / max(residual_rms, 1.0e-300))
    return {
        "transfer_raw": transfer,
        "ir_raw_rolled": ir_rolled,
        "frequencies": frequencies,
        "nfft": nfft,
        "window_starts": tuple(window_starts),
        "ir": ir,
        "time_axis": time_axis,
        "signal_rms": signal_rms,
        "residual_rms": residual_rms,
        "snr_db": float(snr_db),
        "truncation": truncation,
        "origin_shift": origin_shift,
        "truncation_energy_fraction": truncation_energy_fraction,
        "window_leakage_db": window_leakage_db,
        "transfer_relative_noise": transfer_relative_noise,
    }


def estimate_ir_set(run: MeasurementRun, config: WorkbenchConfig) -> DeconvolutionResult:
    """Estimate all channel IRs with one common gain normalization."""

    deconv = config.deconvolution
    raw_results = [
        _raw_channel_estimate(run, channel_index, deconv)
        for channel_index in range(len(run.config.channels))
    ]
    reference_index = int(config.analysis.reference_channel)
    reference_ir = raw_results[reference_index]["ir_raw_rolled"]
    reference_peak = float(np.max(np.abs(reference_ir)))
    if reference_peak <= 0.0:
        fail(ErrorCode.AUDIO_FORMAT, "reference IR has zero energy", stage="deconvolution")

    channel_results: list[IRChannelResult] = []
    for channel_index, raw in enumerate(raw_results):
        ir = np.asarray(raw["ir"], dtype=np.float64) / reference_peak
        transfer = np.asarray(raw["transfer_raw"], dtype=np.complex128) / reference_peak
        absolute = np.abs(ir)
        peak_index = int(np.argmax(absolute))
        peak_delta = 0.0
        if 0 < peak_index < ir.size - 1:
            left, center, right = absolute[peak_index - 1 : peak_index + 2]
            curvature = float(left - 2.0 * center + right)
            if abs(curvature) > 1.0e-30:
                peak_delta = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
        peak_sample = int(raw["time_axis"][peak_index])
        arrival = float(peak_sample + peak_delta)
        peak_level = float(absolute[peak_index])
        peak_db = 20.0 * math.log10(max(peak_level, 1.0e-300))
        channel_results.append(
            IRChannelResult(
                channel_index=channel_index,
                ir=ir,
                time_axis_samples=np.asarray(raw["time_axis"], dtype=np.int64),
                frequencies_hz=np.asarray(raw["frequencies"], dtype=np.float64),
                transfer=transfer,
                peak_index=peak_index,
                peak_sample=peak_sample,
                arrival_sample=arrival,
                peak_delta_samples=peak_delta,
                peak_level_linear=peak_level,
                peak_level_db=peak_db,
                snr_db=float(raw["snr_db"]),
                nfft=int(raw["nfft"]),
                window=deconv.window,
                truncation_samples=int(raw["truncation"]),
                origin_shift_samples=int(raw["origin_shift"]),
                segment_offsets=tuple(int(value) for value in run.sweep_starts),
                window_starts=tuple(int(value) for value in raw["window_starts"]),
                normalization_factor=float(reference_peak),
                residual_rms=float(raw["residual_rms"]),
                signal_rms=float(raw["signal_rms"]),
                transfer_relative_noise=np.asarray(raw["transfer_relative_noise"], dtype=np.float64),
                truncation_energy_fraction=float(raw["truncation_energy_fraction"]),
                window_leakage_db=float(raw["window_leakage_db"]),
                metadata={
                    "regularization": deconv.regularization,
                    "effective_band_hz": [deconv.effective_low_hz, deconv.effective_high_hz],
                    "band_taper_hz": deconv.band_taper_hz,
                    "tukey_alpha": deconv.tukey_alpha,
                    "pre_margin_samples": deconv.pre_margin_samples,
                    "post_margin_samples": deconv.post_margin_samples,
                    "fft_scale": "numpy-rfft",
                    "normalization": "common reference-channel IR peak",
                },
            )
        )

    result_hash = content_hash(
        {
            "deconvolution": deconv,
            "measurement": {
                "source_hash": run.source_hash,
                "recorded_hash": run.recorded_hash,
            },
            "channels": [
                {
                    "ir_hash": channel.ir_hash,
                    "nfft": channel.nfft,
                    "window": channel.window,
                    "truncation": channel.truncation_samples,
                    "origin_shift": channel.origin_shift_samples,
                    "normalization_factor": channel.normalization_factor,
                }
                for channel in channel_results
            ],
        }
    )
    return DeconvolutionResult(
        config=deconv,
        channels=tuple(channel_results),
        nfft=channel_results[0].nfft,
        result_hash=result_hash,
        metadata={
            "layout": {
                "n_sweep": measurement_layout(run.config).n_sweep,
                "sweep_starts": list(run.sweep_starts),
                "window_starts": list(channel_results[0].window_starts),
            }
        },
    )


def recompute_channel_ir(
    run: MeasurementRun,
    config: WorkbenchConfig,
    channel_index: int,
) -> IRChannelResult:
    """Public single-channel recomputation entry point used for auditability."""

    if not (0 <= channel_index < len(run.config.channels)):
        fail(
            ErrorCode.BAND_INVALID,
            "channel index is outside the recording",
            stage="deconvolution",
            details={"channel_index": channel_index},
        )
    raw_all = _raw_channel_estimate(
        run,
        int(config.analysis.reference_channel),
        config.deconvolution,
    )
    reference_peak = float(np.max(np.abs(raw_all["ir_raw_rolled"])))
    if channel_index == int(config.analysis.reference_channel):
        raw = raw_all
    else:
        raw = _raw_channel_estimate(run, channel_index, config.deconvolution)
    ir = np.asarray(raw["ir"], dtype=np.float64) / reference_peak
    transfer = np.asarray(raw["transfer_raw"], dtype=np.complex128) / reference_peak
    absolute = np.abs(ir)
    peak_index = int(np.argmax(absolute))
    peak_delta = 0.0
    if 0 < peak_index < ir.size - 1:
        left, center, right = absolute[peak_index - 1 : peak_index + 2]
        curvature = float(left - 2.0 * center + right)
        if abs(curvature) > 1.0e-30:
            peak_delta = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    peak_sample = int(raw["time_axis"][peak_index])
    return IRChannelResult(
        channel_index=channel_index,
        ir=ir,
        time_axis_samples=np.asarray(raw["time_axis"], dtype=np.int64),
        frequencies_hz=np.asarray(raw["frequencies"], dtype=np.float64),
        transfer=transfer,
        peak_index=peak_index,
        peak_sample=peak_sample,
        arrival_sample=float(peak_sample + peak_delta),
        peak_delta_samples=peak_delta,
        peak_level_linear=float(absolute[peak_index]),
        peak_level_db=20.0 * math.log10(max(float(absolute[peak_index]), 1.0e-300)),
        snr_db=float(raw["snr_db"]),
        nfft=int(raw["nfft"]),
        window=config.deconvolution.window,
        truncation_samples=int(raw["truncation"]),
        origin_shift_samples=int(raw["origin_shift"]),
        segment_offsets=tuple(int(value) for value in run.sweep_starts),
        window_starts=tuple(int(value) for value in raw["window_starts"]),
        normalization_factor=reference_peak,
        residual_rms=float(raw["residual_rms"]),
        signal_rms=float(raw["signal_rms"]),
        transfer_relative_noise=np.asarray(raw["transfer_relative_noise"], dtype=np.float64),
        truncation_energy_fraction=float(raw["truncation_energy_fraction"]),
        window_leakage_db=float(raw["window_leakage_db"]),
        metadata={
            "regularization": config.deconvolution.regularization,
            "effective_band_hz": [
                config.deconvolution.effective_low_hz,
                config.deconvolution.effective_high_hz,
            ],
            "band_taper_hz": config.deconvolution.band_taper_hz,
            "tukey_alpha": config.deconvolution.tukey_alpha,
            "pre_margin_samples": config.deconvolution.pre_margin_samples,
            "post_margin_samples": config.deconvolution.post_margin_samples,
        },
    )
