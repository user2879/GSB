"""Deterministic acoustic analysis, fixed-point mapping and frame synthesis.

All accumulation happens in an explicit, order-stable sequence so that a thread
pool and a single-threaded executor produce byte-identical PCM.  No global
random state is used; unvoiced excitation derives from a per-segment PCG.
"""
from __future__ import annotations

import numpy as np

from .config import PipelineConfig
from .errors import E_SEGMENT_TOO_LONG, FailureLog
from .model import Segment


def to_fixed(value: float, frac_bits: int) -> int:
    """Round half away from zero into a signed fixed-point integer."""
    scale = 1 << frac_bits
    if value >= 0:
        return int(value * scale + 0.5)
    return -int(-value * scale + 0.5)


def from_fixed(value: int, frac_bits: int) -> float:
    return value / (1 << frac_bits)


def snap_grid(value: int, levels: int) -> int:
    """Snap a fixed-point value onto ``levels`` symmetric grid steps."""
    if levels <= 1:
        return value
    if value == 0:
        return 0
    if value > 0:
        step = max(1, int(value / levels + 0.5))
    else:
        step = -max(1, int(-value / levels + 0.5))
    return int(step * levels)


def frame_centers(frame_count: int, frame_length: int,
                  frame_shift: int) -> np.ndarray:
    return (np.arange(frame_count, dtype=np.int64) * frame_shift
            + frame_length // 2)


def extract_frames(pcm: np.ndarray, seg: Segment, config: PipelineConfig
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (frames, centers) for the integer frame span of one segment.

    Only the segment's own source samples are ever read.
    """
    start = int(round(seg.start_frame))
    end = int(round(seg.end_frame))
    count = end - start
    fl, fs = config.frame_length, config.frame_shift
    centers = start * fs + fl // 2 + np.arange(count, dtype=np.int64) * fs
    frames = np.zeros((count, fl), dtype=np.float64)
    half = fl // 2
    for idx, center in enumerate(centers):
        left, right = center - half, center - half + fl
        lo, hi = max(0, left), min(len(pcm), right)
        if hi > lo:
            frames[idx, lo - left:hi - left] = pcm[lo:hi].astype(np.float64)
    return frames, centers


def _mel_filterbank(config: PipelineConfig) -> np.ndarray:
    fl = config.frame_length
    fft_bins = fl // 2 + 1
    fmax = min(config.mel_fmax, config.sample_rate / 2.0)

    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(fmax), config.mel_bands + 2)
    points = np.array([mel_to_hz(m) for m in mels])
    bin_hz = np.arange(fft_bins) * config.sample_rate / fl
    lo = points[:-2, None]
    center = points[1:-1, None]
    hi = points[2:, None]
    hz = bin_hz[None, :]
    rising = (hz >= lo) & (hz <= center) & (center > lo)
    falling = (hz > center) & (hz <= hi) & (hi > center)
    bank = np.where(
        rising, (hz - lo) / np.where(center > lo, center - lo, 1.0),
        np.where(falling, (hi - hz) / np.where(hi > center, hi - center, 1.0),
                 0.0),
    )
    return bank


def _frame_f0(frame: np.ndarray, config: PipelineConfig) -> float:
    """Autocorrelation pitch estimate inside the configured F0 search range."""
    fl = config.frame_length
    min_lag = max(1, int(config.sample_rate / config.f0_max_hz))
    max_lag = min(fl - 1, int(config.sample_rate / config.f0_min_hz))
    centered = frame - frame.mean()
    energy = float(np.dot(centered, centered))
    if energy <= 1e-9:
        return 0.0
    corr = np.correlate(centered, centered, mode="full")[fl - 1:]
    zero_lag = corr[0]
    if zero_lag <= 0:
        return 0.0
    search = corr[min_lag:max_lag + 1]
    lag = int(np.argmax(search)) + min_lag
    strength = float(corr[lag]) / float(zero_lag)
    if strength < 0.35:
        return 0.0
    return config.sample_rate / float(lag)


def _frame_cepstrum(frame: np.ndarray, order: int) -> np.ndarray:
    spectrum = np.fft.rfft(frame * np.hanning(len(frame)))
    log_mag = np.log(np.abs(spectrum) + 1e-8)
    cepstrum = np.fft.irfft(log_mag)
    return cepstrum[:order].copy()


def _frame_mel(frame: np.ndarray, bank: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    return np.log(bank @ spectrum + 1e-8)


def analyze_segment(pcm: np.ndarray, seg: Segment, config: PipelineConfig,
                    failures: FailureLog) -> dict | None:
    """F0 / cepstrum / energy statistics aggregated in strict frame order."""
    count = int(round(seg.end_frame)) - int(round(seg.start_frame))
    if count <= 0:
        failures.add("E_EMPTY_SEGMENT", "analysis", segment=seg.index,
                     field="frames", message="no frames in segment")
        return None
    if count > config.max_frames_per_segment:
        failures.add(E_SEGMENT_TOO_LONG, "analysis", segment=seg.index,
                     field="frames",
                     message=f"{count} > {config.max_frames_per_segment}")
        return None
    frames, centers = extract_frames(pcm, seg, config)
    bank = _mel_filterbank(config)
    order = config.cepstrum_order
    f0_list, rms_list, voiced_flags = [], [], []
    cep_sum = np.zeros(order, dtype=np.float64)
    mel_sum = np.zeros(config.mel_bands, dtype=np.float64)
    for frame in frames:  # explicit ascending-order accumulation
        f0 = _frame_f0(frame, config)
        rms = float(np.sqrt(np.dot(frame, frame) / len(frame) + 1e-12))
        cep = _frame_cepstrum(frame, order)
        mel = _frame_mel(frame, bank)
        f0_list.append(f0)
        rms_list.append(rms)
        voiced_flags.append(f0 > 0.0)
        cep_sum += cep
        mel_sum += mel
    voiced_f0 = [v for v in f0_list if v > 0.0]
    return {
        "segment": seg.index,
        "frame_count": count,
        "frame_centers": [int(c) for c in centers],
        "source_frame_set": list(range(int(round(seg.start_frame)),
                                       int(round(seg.end_frame)))),
        "f0_hz": [float(v) for v in f0_list],
        "rms": rms_list,
        "voiced": voiced_flags,
        "mean_cepstrum": (cep_sum / count).tolist(),
        "mean_mel": (mel_sum / count).tolist(),
        "mean_f0_hz": (sum(voiced_f0) / len(voiced_f0)) if voiced_f0 else 0.0,
        "mean_rms": sum(rms_list) / len(rms_list),
    }


def map_segment(analysis: dict, seg: Segment,
                config: PipelineConfig) -> dict:
    """Deterministic fixed-point timbre shift/bend of F0 and cepstrum."""
    entry = config.timbre_entry(seg.timbre)
    if entry is None:
        raise KeyError(f"unknown timbre {seg.timbre!r}")
    frac = config.frac_bits
    levels = config.quant_levels
    intensity_q = snap_grid(to_fixed(seg.intensity, frac), levels)
    f0_shift_q = snap_grid(
        to_fixed(float(entry["f0_shift_ratio"]), frac), levels)
    formant_shift_q = snap_grid(
        to_fixed(float(entry["formant_shift_ratio"]), frac), levels)
    bend_q = snap_grid(to_fixed(float(entry["cep_bend"]), frac), levels)

    mapped_f0, mapped_cep = [], []
    cep_order = len(analysis["mean_cepstrum"])
    one = 1 << frac
    for f0 in analysis["f0_hz"]:
        if f0 <= 0.0:
            mapped_f0.append(0)
            continue
        base_q = to_fixed(f0, frac)
        # Q-fixed multiply: base * (1 + intensity * f0_shift), then grid snap.
        factor = one + (intensity_q * f0_shift_q) // one
        mapped = snap_grid((base_q * factor) // one, levels)
        mapped_f0.append(mapped)
    for order, value in enumerate(analysis["mean_cepstrum"]):
        base_q = to_fixed(value, frac)
        shift = (formant_shift_q * intensity_q) // one
        bend = (bend_q * intensity_q * order) // one
        mapped_cep.append(snap_grid(base_q + shift + bend, levels))
    gain_factor = one + (
        intensity_q * to_fixed(float(entry["gain_delta"]), frac)) // one
    return {
        "segment": seg.index,
        "timbre": seg.timbre,
        "quant_levels": levels,
        "frac_bits": frac,
        "intensity_q": intensity_q,
        "f0_shift_q": f0_shift_q,
        "formant_shift_q": formant_shift_q,
        "bend_q": bend_q,
        "gain_factor_q": gain_factor,
        "mapped_f0_q": mapped_f0,
        "mapped_cepstrum_q": mapped_cep,
        "cepstrum_order": cep_order,
        "rms": analysis["rms"],
        "voiced": analysis["voiced"],
        "source_frame_set": analysis["source_frame_set"],
        "parameter_set": [
            intensity_q, f0_shift_q, formant_shift_q, bend_q,
            gain_factor, levels,
        ],
    }


class _DeterministicNoise:
    """32-bit PCG producing per-segment, per-frame reproducible noise."""

    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        self.inc = 0x9E3779B9

    def next_float(self) -> float:
        self.state = (self.state * 1664525 + self.inc) & 0xFFFFFFFF
        # xorshift mixing keeps the excitation white and reproducible
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        return ((x & 0xFFFFFFFF) / 2147483648.0) - 1.0


def _envelope(cepstrum_q: list[int], frac: int, bins: int) -> np.ndarray:
    log_mag = np.fft.rfft(np.asarray(cepstrum_q, dtype=np.float64) /
                          (1 << frac), n=2 * (bins - 1))
    return np.exp(np.real(log_mag[:bins]))


def synthesize_segment(analysis: dict, mapping: dict, seg: Segment,
                       config: PipelineConfig) -> np.ndarray:
    """Overlap-add one segment using only its own frames and mapping."""
    fl, fs = config.frame_length, config.frame_shift
    count = analysis["frame_count"]
    length = (count - 1) * fs + fl
    buffer = np.zeros(length, dtype=np.float64)
    window = np.hanning(fl).astype(np.float64)
    bins = fl // 2 + 1
    envelope = _envelope(mapping["mapped_cepstrum_q"], config.frac_bits, bins)
    rng = _DeterministicNoise(seed=0x51ED0000 ^ (seg.index * 0x01000193))
    one = 1 << config.frac_bits
    gain = mapping["gain_factor_q"] / one

    for frame_idx in range(count):
        f0_q = mapping["mapped_f0_q"][frame_idx]
        frame = np.zeros(fl, dtype=np.float64)
        if f0_q > 0 and analysis["voiced"][frame_idx]:
            f0 = f0_q / one
            harmonic = 1
            while harmonic * f0 < config.sample_rate / 2.0:
                target = harmonic * f0
                bin_idx = int(round(target / (config.sample_rate / fl)))
                if bin_idx >= bins:
                    break
                amp = envelope[bin_idx]
                phase = (2.0 * np.pi * target *
                         np.arange(fl) / config.sample_rate)
                frame += amp * np.cos(phase)
                harmonic += 1
        else:
            for sample in range(fl):
                frame[sample] = rng.next_float() * envelope.mean()
        rms = max(analysis["rms"][frame_idx], 1e-6)
        frame = frame / (np.sqrt(np.dot(frame, frame) / fl) + 1e-9) * rms
        offset = frame_idx * fs
        buffer[offset:offset + fl] += frame * window * gain
    return np.clip(np.rint(buffer), -32768, 32767).astype("<i2")
