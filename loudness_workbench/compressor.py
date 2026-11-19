"""Log-domain dynamic-range compressor with persistent smoothing state.

The control plane runs on a fixed 10 ms hop that is independent of the audio
block length. One-pole attack/release smoothing is applied in the dB domain,
which makes the gain envelope bit-identical whether the signal is fed as one
large offline buffer or as many small real-time chunks, and guarantees no
state reset when the UI block size or processing mode changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    CHANNEL_WEIGHTS_DB,
    CONTROL_HOP_MS,
    SAMPLE_RATE,
    CompressorConfig,
    ms_to_samples,
)
from .errors import E_COMP_STATE, WorkbenchError

EPS = 1e-12


def _time_constant_alpha(tau_ms: float, hop_ms: float) -> float:
    return float(1.0 - np.exp(-hop_ms / tau_ms))


@dataclass(frozen=True)
class CompressorFrame:
    index: int
    start_sample: int
    end_sample: int
    input_level_dbfs: float
    target_gain_db: float
    smoothed_gain_db: float


class LogDomainCompressor:
    """Sample-rate independent compressor operating on fixed control hops."""

    def __init__(self, config: CompressorConfig, sample_rate: int = SAMPLE_RATE,
                 hop_ms: int = CONTROL_HOP_MS) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self.hop_samples = ms_to_samples(hop_ms, sample_rate)
        if self.hop_samples <= 0:
            raise WorkbenchError(E_COMP_STATE, "compressor control hop must be positive")
        self.attack_alpha = _time_constant_alpha(config.attack_ms, hop_ms)
        self.release_alpha = _time_constant_alpha(config.release_ms, hop_ms)
        weights = np.asarray(
            [0.0 if np.isinf(w) else 10.0 ** (w / 10.0) for w in CHANNEL_WEIGHTS_DB]
        )
        self._weights = weights / weights.sum()
        self._pending = np.empty((0, len(weights)))
        self._cursor = 0
        self.envelope_db = 0.0  # smoothed gain, 0 dB means unity
        self.frames: list[CompressorFrame] = []

    def _static_gain_db(self, level_dbfs: float) -> float:
        cfg = self.config
        knee = cfg.knee_db
        over = level_dbfs - cfg.threshold_dbfs
        if over <= -knee / 2.0:
            reduction = 0.0
        elif abs(over) < knee / 2.0:
            reduction = (1.0 / cfg.ratio - 1.0) * (over + knee / 2.0) ** 2 / (2.0 * knee)
        else:
            reduction = (1.0 / cfg.ratio - 1.0) * over
        return float(cfg.makeup_gain_db + reduction)

    def _run_frame(self, frame: np.ndarray) -> CompressorFrame:
        rms = float(np.sqrt(np.mean(frame**2, axis=0) @ self._weights))
        level = 20.0 * np.log10(max(rms, EPS))
        target = self._static_gain_db(level)
        alpha = self.attack_alpha if target < self.envelope_db else self.release_alpha
        self.envelope_db = (1.0 - alpha) * self.envelope_db + alpha * target
        start = self._cursor
        end = start + frame.shape[0]
        record = CompressorFrame(
            index=len(self.frames),
            start_sample=start,
            end_sample=end,
            input_level_dbfs=level,
            target_gain_db=target,
            smoothed_gain_db=self.envelope_db,
        )
        self.frames.append(record)
        self._cursor = end
        return record

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Process an arbitrary-length chunk and return the gain-scaled audio."""
        if chunk.ndim != 2 or chunk.shape[1] != len(self._weights):
            raise WorkbenchError(
                E_COMP_STATE,
                f"compressor chunk shape {chunk.shape} incompatible with channel weights",
            )
        buffer = np.concatenate([self._pending, chunk], axis=0)
        full_frames = buffer.shape[0] // self.hop_samples
        used = full_frames * self.hop_samples
        for k in range(full_frames):
            self._run_frame(buffer[k * self.hop_samples : (k + 1) * self.hop_samples])
        self._pending = buffer[used:]
        gains = np.asarray([f.smoothed_gain_db for f in self.frames[-full_frames:]])
        linear = 10.0 ** (gains / 20.0)
        repeated = np.repeat(linear, self.hop_samples)
        output = buffer[:used] * repeated[:, None]
        return output

    def gain_envelope(self) -> np.ndarray:
        return np.asarray([f.smoothed_gain_db for f in self.frames], dtype=np.float64)

    def frame_increments(self) -> np.ndarray:
        """Natural per-hop smoothing increments of the gain envelope."""
        env = self.gain_envelope()
        if env.size < 2:
            return np.zeros(max(env.size - 1, 0))
        return np.abs(np.diff(env))

    def gain_at_sample(self, sample_index: int) -> float:
        """Smoothed gain active at an arbitrary sample position."""
        frame_index = min(sample_index // self.hop_samples, len(self.frames) - 1)
        if frame_index < 0:
            return 0.0
        return self.frames[frame_index].smoothed_gain_db

    def get_state(self) -> dict:
        return {
            "envelope_db": self.envelope_db,
            "cursor": self._cursor,
            "pending": self._pending.copy(),
            "frames": [f.__dict__.copy() for f in self.frames],
            "config": self.config.__dict__,
            "hop_samples": self.hop_samples,
        }

    def set_state(self, state: dict) -> None:
        expected = self.config.__dict__
        if state.get("config") != expected:
            raise WorkbenchError(
                E_COMP_STATE,
                "serialized compressor parameters do not match the active configuration",
            )
        if state.get("hop_samples") != self.hop_samples:
            raise WorkbenchError(
                E_COMP_STATE,
                f"hop mismatch: state={state.get('hop_samples')} active={self.hop_samples}",
            )
        self.envelope_db = float(state["envelope_db"])
        self._cursor = int(state["cursor"])
        self._pending = np.asarray(state["pending"]).copy()
        self.frames = [CompressorFrame(**item) for item in state["frames"]]
