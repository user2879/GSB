"""Frame-level acoustic features and character emission scoring."""

import math

import numpy as np

from .audio import SAMPLE_RATE
from .errors import E_CONFIG_INVALID, PipelineError


def frame_signal(samples, frame_length_ms, frame_shift_ms):
    """Split samples into frames; the final short frame is zero-padded."""
    length = SAMPLE_RATE * frame_length_ms // 1000
    shift = SAMPLE_RATE * frame_shift_ms // 1000
    n_samples = samples.shape[0]
    if n_samples == 0:
        return np.zeros((0, length), dtype=np.float64), np.zeros((0,), dtype=bool)
    n_frames = 1 + int(math.ceil(max(0, n_samples - length) / shift))
    frames = np.zeros((n_frames, length), dtype=np.float64)
    padded = np.zeros(length, dtype=np.float64)
    last_padded = np.zeros(n_frames, dtype=bool)
    for idx in range(n_frames):
        start = idx * shift
        end = min(n_samples, start + length)
        chunk = samples[start:end]
        if end - start < length:
            padded[: chunk.shape[0]] = chunk
            frames[idx] = padded
            padded.fill(0.0)
            last_padded[idx] = True
        else:
            frames[idx] = chunk
    return frames, last_padded


def mel_filterbank(n_bands, fft_size):
    """Deterministic triangular Mel filterbank (no external deps)."""
    low_hz, high_hz = 80.0, 7600.0

    def hz_to_mel(hz):
        return 1127.0 * math.log1p(hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (math.exp(mel / 1127.0) - 1.0)

    low_mel = hz_to_mel(low_hz)
    high_mel = hz_to_mel(high_hz)
    points = [mel_to_hz(low_mel + (high_mel - low_mel) * i / (n_bands + 1))
              for i in range(n_bands + 2)]
    bin_hz = np.arange(fft_size // 2 + 1, dtype=np.float64) * (SAMPLE_RATE / fft_size)
    bank = np.zeros((n_bands, fft_size // 2 + 1), dtype=np.float64)
    for band in range(n_bands):
        left, center, right = points[band], points[band + 1], points[band + 2]
        rising = (bin_hz - left) / (center - left)
        falling = (right - bin_hz) / (right - center)
        left_half = (bin_hz >= left) & (bin_hz <= center)
        right_half = (bin_hz > center) & (bin_hz <= right)
        bank[band] = np.where(left_half, rising, np.where(right_half, falling, 0.0))
    return bank


def quantize(values, mode, frac_bits):
    """Fixed-point quantization applied in the log-emission domain."""
    if mode == "fp64":
        return values
    scale = float(2 ** frac_bits)
    qmax = 2 ** (31 if mode == "q32" else 15) - 1
    qmin = -(qmax + 1)
    scaled = np.clip(np.rint(values * scale), qmin, qmax) / scale
    return scaled.astype(np.float64)


def _round_significant(value, digits):
    if value == 0.0 or not math.isfinite(value):
        return float(value)
    magnitude = math.floor(math.log10(abs(value)))
    factor = 10.0 ** (digits - 1 - magnitude)
    return math.floor(value * factor + 0.5) / factor if value >= 0 else \
        -math.floor(-value * factor + 0.5) / factor


def _round_significant_array(values, digits):
    result = np.array(values, dtype=np.float64, copy=True)
    finite = np.isfinite(result)
    nonzero = finite & (result != 0.0)
    magnitude = np.floor(np.log10(np.abs(result[nonzero])))
    factor = 10.0 ** (digits - 1 - magnitude)
    scaled = result[nonzero] * factor
    result[nonzero] = np.where(
        result[nonzero] >= 0,
        np.floor(scaled + 0.5),
        -np.floor(-scaled + 0.5),
    ) / factor
    return result


def compute_emissions(samples, score_params, schedule_config, injection=None):
    """Return per-frame emission matrix, frame flags and per-frame point costs."""
    frame_length_ms = score_params["frame_length_ms"]
    frame_shift_ms = score_params["frame_shift_ms"]
    vocab = score_params["vocab"]
    n_vocab = len(vocab)
    mel_bands = score_params["mel_bands"]
    table = score_params["emission_table"]
    smooth = score_params["log_smooth"]
    digits = score_params["significant_digits"]

    if injection and injection["type"] == "emission_dim_mismatch":
        bad_table = [row[: max(0, len(row) - 1)] for row in table]
        if len(bad_table[0]) != n_vocab:
            raise PipelineError(
                E_CONFIG_INVALID,
                f"emission_table width {len(bad_table[0])} != vocab size {n_vocab}",
                stage="feature", injection_id=injection["id"],
            )

    frames, last_padded = frame_signal(samples, frame_length_ms, frame_shift_ms)
    n_frames = frames.shape[0]
    fft_size = 1
    while fft_size < frames.shape[1]:
        fft_size <<= 1
    bank = mel_filterbank(mel_bands, fft_size)

    emissions = np.full((n_frames, n_vocab), float(smooth), dtype=np.float64)
    if n_frames:
        windowed = frames * np.hanning(frames.shape[1])
        spectrum = np.abs(np.fft.rfft(windowed, n=fft_size, axis=1)) ** 2
        mel_energy = np.maximum(spectrum @ bank.T, 1e-12)
        log_mel = np.log(mel_energy)
        dominant_band = np.argmax(log_mel, axis=1)
        for frame_idx in range(n_frames):
            band = int(dominant_band[frame_idx])
            emissions[frame_idx] = np.asarray(table[band], dtype=np.float64)

    emissions = quantize(
        emissions, schedule_config["quant_mode"], schedule_config["quant_frac_bits"]
    )
    emissions = _round_significant_array(emissions, digits)

    point_costs = np.full((n_frames,), n_vocab, dtype=np.int64)
    return {
        "emissions": emissions,
        "last_padded": last_padded,
        "point_costs": point_costs,
        "frame_count": int(n_frames),
    }
