"""ITU-R BS.1770-4 loudness measurement, true-peak and loudness range.

The same :class:`StreamingLoudnessMeter` drives both the block-wise real-time
path and the offline full-scan path. K-weighting filter state and the 400 ms
integration window are carried across calls, so chunk boundaries cannot change
the accumulated statistics. Offline analysis simply feeds the whole file into
one meter instance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .config import (
    CHANNEL_WEIGHTS_DB,
    LRA_ABSOLUTE_THRESHOLD_LKFS,
    LRA_HIGH_PERCENTILE,
    LRA_LOW_PERCENTILE,
    SAMPLE_RATE,
    TRUE_PEAK_UPSAMPLE,
    ms_to_samples,
)

# BS.1770-4 stage-1 high-shelf (gain +4 dB above ~1.68 kHz).
SHELF_V = 10.0 ** (4.0 / 20.0)
SHELF_FC = 1681.974450955533
SHELF_Q = 0.7071752369554196
# BS.1770-4 stage-2 RLB high-pass filter.
RLB_FC = 38.13547087602444
RLB_Q = 0.5003270373238773


def k_weighting_coeffs(sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Return the combined second-order K-weighting ``(b, a)`` coefficients."""
    k = 2.0 * np.tan(np.pi * SHELF_FC / sample_rate)
    vb = SHELF_V
    vb2 = vb**2
    q = SHELF_Q
    denom = k**2 * vb2 + k * (vb / q) + 1.0
    b_hs = np.array(
        [
            (vb2 * k**2 + vb * (k / q) + 1.0) / denom,
            (2.0 * (vb2 * k**2 - 1.0)) / denom,
            (vb2 * k**2 - vb * (k / q) + 1.0) / denom,
        ]
    )
    a_hs = np.array([
        1.0,
        (2.0 * (vb2 * k**2 - 1.0)) / denom,
        (vb2 * k**2 - vb * (k / q) + 1.0) / denom,
    ])
    k_rlb = 2.0 * np.tan(np.pi * RLB_FC / sample_rate)
    q_rlb = RLB_Q
    b_rlb = np.array([1.0, -2.0, 1.0])
    a_rlb = np.array(
        [
            1.0,
            (2.0 * (k_rlb**2) - 2.0) / (k_rlb**2 + k_rlb / q_rlb + 1.0),
            (k_rlb**2 - k_rlb / q_rlb + 1.0) / (k_rlb**2 + k_rlb / q_rlb + 1.0),
        ]
    )
    b = np.polymul(b_hs, b_rlb)
    a = np.polymul(a_hs, a_rlb)
    return b, a


def channel_gains_db() -> np.ndarray:
    return np.asarray(
        [0.0 if np.isinf(w) else w for w in CHANNEL_WEIGHTS_DB], dtype=np.float64
    )


def lkfs_from_mean_square(ms: float) -> float:
    if ms <= 0.0:
        return float("-inf")
    return float(-0.691 + 10.0 * np.log10(ms))


@dataclass(frozen=True)
class BlockStat:
    """One 400 ms window statistic (also the short-term loudness series)."""

    index: int
    start_sample: int
    end_sample: int
    start_s: float
    end_s: float
    loudness_lkfs: float
    mean_square: float


@dataclass(frozen=True)
class GatedResult:
    integrated_lkfs: float
    absolute_gate_lkfs: float
    relative_gate_lkfs: float
    blocks_total: int
    blocks_below_absolute: int
    blocks_below_relative: int
    blocks_kept: int
    lra_lu: float
    short_term: tuple[BlockStat, ...]


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile, method="linear"))


def gated_loudness(
    blocks: list[BlockStat],
    absolute_gate: float,
    relative_offset: float,
) -> GatedResult:
    """Apply the BS.1770 two-level gate and compute LRA."""
    total = len(blocks)
    finite = [b for b in blocks if np.isfinite(b.loudness_lkfs)]
    abs_pass = [b for b in finite if b.loudness_lkfs >= absolute_gate]
    below_abs = total - len(abs_pass)
    if not abs_pass:
        return GatedResult(
            float("-inf"), absolute_gate, float("-inf"), total,
            below_abs, 0, 0, float("-inf"), tuple(blocks),
        )
    mean_abs = float(
        -0.691 + 10.0 * np.log10(np.mean([b.mean_square for b in abs_pass]))
    )
    relative_gate = mean_abs + relative_offset
    rel_pass = [b for b in abs_pass if b.loudness_lkfs > relative_gate]
    if not rel_pass:
        rel_pass = abs_pass
        relative_gate = absolute_gate
    integrated = lkfs_from_mean_square(float(np.mean([b.mean_square for b in rel_pass])))
    below_rel = len(abs_pass) - len(rel_pass)
    lra = _loudness_range(abs_pass)
    return GatedResult(
        integrated, absolute_gate, relative_gate, total,
        below_abs, below_rel, len(rel_pass), lra, tuple(blocks),
    )


def _loudness_range(abs_pass: list[BlockStat]) -> float:
    """BS.1770-4 loudness range from the -70 LKFS absolute-gated blocks."""
    values = np.array([b.loudness_lkfs for b in abs_pass if b.loudness_lkfs >= LRA_ABSOLUTE_THRESHOLD_LKFS])
    if values.size < 2:
        return float("-inf")
    low = _percentile(values, LRA_LOW_PERCENTILE)
    high = _percentile(values, LRA_HIGH_PERCENTILE)
    return float(high - low)


def design_upsample_filter(upsample: int = TRUE_PEAK_UPSAMPLE, taps: int = 41) -> np.ndarray:
    """Deterministic linear-phase anti-imaging FIR for true-peak upsampling."""
    cutoff = 1.0 / upsample
    coeffs = sps.firwin(taps, cutoff, window="hamming")
    coeffs = coeffs * upsample  # unit pass-band gain after zero insertion
    return coeffs


class TruePeakDetector:
    """Stateful 4x oversampling true-peak detector.

    FIR filter state is carried across calls so streaming chunks give the same
    peak as the offline full-length signal (reported within the declared
    tolerance; the offline value remains the reference for the report).
    """

    def __init__(self, channels: int, upsample: int = TRUE_PEAK_UPSAMPLE,
                 sample_rate: int = SAMPLE_RATE) -> None:
        self.channels = channels
        self.upsample = upsample
        self.sample_rate = sample_rate
        self.fir = design_upsample_filter(upsample)
        self._state = np.zeros((channels, self.fir.size - 1))
        self.peak_linear = 0.0
        self._input_samples = 0

    def process(self, chunk: np.ndarray) -> float:
        n = chunk.shape[0]
        inserted = np.zeros((n * self.upsample, self.channels))
        inserted[:: self.upsample, :] = chunk
        filtered = np.empty_like(inserted)
        for ch in range(self.channels):
            filtered[:, ch], self._state[ch] = sps.lfilter(
                self.fir, 1.0, inserted[:, ch], zi=self._state[ch]
            )
        peak = float(np.max(np.abs(filtered))) if filtered.size else 0.0
        self.peak_linear = max(self.peak_linear, peak)
        self._input_samples += n
        return self.peak_dbtp()

    def peak_dbtp(self) -> float:
        if self.peak_linear <= 0.0:
            return float("-inf")
        return float(20.0 * np.log10(self.peak_linear))

    def get_state(self) -> dict:
        return {
            "fir_state": self._state.copy(),
            "peak_linear": self.peak_linear,
            "input_samples": self._input_samples,
        }

    def set_state(self, state: dict) -> None:
        self._state = np.asarray(state["fir_state"]).copy()
        self.peak_linear = float(state["peak_linear"])
        self._input_samples = int(state["input_samples"])


class StreamingLoudnessMeter:
    """Block-streaming BS.1770 meter with persistent filter/window state."""

    def __init__(self, sample_rate: int = SAMPLE_RATE,
                 window_ms: int = 400, hop_ms: int = 100) -> None:
        self.sample_rate = sample_rate
        self.window_samples = ms_to_samples(window_ms, sample_rate)
        self.hop_samples = ms_to_samples(hop_ms, sample_rate)
        b, a = k_weighting_coeffs(sample_rate)
        self.b, self.a = b, a
        self.channels = len(channel_gains_db())
        self._filter_state = sps.lfilter_zi(b, a)[:, None] * np.zeros(
            (1, self.channels)
        )
        gains = channel_gains_db()
        self._gains = 10.0 ** (gains / 10.0)
        self._gains[np.isneginf(gains)] = 0.0
        self._residual = np.empty((0, self.channels))
        self._sample_cursor = 0
        self.blocks: list[BlockStat] = []
        self.true_peak = TruePeakDetector(self.channels, sample_rate=sample_rate)

    def process(self, chunk: np.ndarray) -> list[BlockStat]:
        """Feed one audio chunk; return the window blocks completed inside it."""
        if chunk.ndim != 2 or chunk.shape[1] != self.channels:
            from .errors import WorkbenchError, E_AUDIO_FORMAT
            raise WorkbenchError(
                E_AUDIO_FORMAT,
                f"chunk shape {chunk.shape} incompatible with {self.channels} channels",
            )
        weighted, self._filter_state = sps.lfilter(
            self.b, self.a, chunk, axis=0, zi=self._filter_state
        )
        self.true_peak.process(chunk)
        buffer = np.concatenate([self._residual, weighted], axis=0)
        completed: list[BlockStat] = []
        start_offset = self._sample_cursor - self._residual.shape[0]
        usable = ((buffer.shape[0] - self.window_samples) // self.hop_samples) + 1
        for k in range(max(usable, 0)):
            w0 = k * self.hop_samples
            window = buffer[w0 : w0 + self.window_samples, :]
            ms = float(np.mean(window**2, axis=0) @ self._gains)
            abs_start = start_offset + w0
            abs_end = abs_start + self.window_samples
            stat = BlockStat(
                index=len(self.blocks),
                start_sample=int(abs_start),
                end_sample=int(abs_end),
                start_s=abs_start / self.sample_rate,
                end_s=abs_end / self.sample_rate,
                loudness_lkfs=lkfs_from_mean_square(ms),
                mean_square=ms,
            )
            self.blocks.append(stat)
            completed.append(stat)
        keep = buffer.shape[0] - max(usable, 0) * self.hop_samples
        self._residual = buffer[-keep:] if keep > 0 else np.empty((0, self.channels))
        self._sample_cursor += chunk.shape[0]
        return completed

    def finalize(self, absolute_gate: float, relative_offset: float) -> GatedResult:
        return gated_loudness(self.blocks, absolute_gate, relative_offset)

    def get_state(self) -> dict:
        return {
            "filter_state": self._filter_state.copy(),
            "residual": self._residual.copy(),
            "sample_cursor": self._sample_cursor,
            "blocks": [b.__dict__.copy() for b in self.blocks],
            "true_peak": self.true_peak.get_state(),
        }

    def set_state(self, state: dict) -> None:
        self._filter_state = np.asarray(state["filter_state"]).copy()
        self._residual = np.asarray(state["residual"]).copy()
        self._sample_cursor = int(state["sample_cursor"])
        self.blocks = [BlockStat(**item) for item in state["blocks"]]
        self.true_peak.set_state(state["true_peak"])
