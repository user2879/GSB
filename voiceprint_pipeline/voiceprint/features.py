"""Fixed-grid frame acoustic features: F0, mel-band energies, cepstrum, energy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio import WaveData
from .config import PipelineConfig
from .util import fingerprint, round_sigfigs, sha256_text

SAMPLE_RATE = 16000
_F0_MIN_HZ = 70.0
_F0_MAX_HZ = 400.0
_EPS = 1e-12


@dataclass
class FrameFeatures:
    sample_id: str
    n_frames: int
    frame_size: int
    hop_size: int
    f0: np.ndarray              # (n_frames,)
    mel_energy: np.ndarray      # (n_frames, mel_bands)
    cepstrum: np.ndarray        # (n_frames, cepstrum_order)
    energy_db: np.ndarray       # (n_frames,)
    is_tail: np.ndarray         # (n_frames,) bool
    feature_fp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "n_frames": self.n_frames,
            "frame_size": self.frame_size,
            "hop_size": self.hop_size,
            "f0": self.f0.tolist(),
            "mel_energy": self.mel_energy.tolist(),
            "cepstrum": self.cepstrum.tolist(),
            "energy_db": self.energy_db.tolist(),
            "is_tail": [bool(x) for x in self.is_tail],
            "feature_fp": self.feature_fp,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrameFeatures":
        return cls(
            sample_id=payload["sample_id"],
            n_frames=int(payload["n_frames"]),
            frame_size=int(payload["frame_size"]),
            hop_size=int(payload["hop_size"]),
            f0=np.array(payload["f0"], dtype=np.float64),
            mel_energy=np.array(payload["mel_energy"], dtype=np.float64),
            cepstrum=np.array(payload["cepstrum"], dtype=np.float64),
            energy_db=np.array(payload["energy_db"], dtype=np.float64),
            is_tail=np.array(payload["is_tail"], dtype=bool),
            feature_fp=payload["feature_fp"],
        )


def frame_grid(n_samples: int, frame_size: int, hop_size: int) -> tuple[int, list[int], list[int]]:
    """Return frame count and start/end sample indices; the final partial frame zero-pads."""
    if n_samples <= 0:
        return 0, [], []
    # A frame start at/after the signal end would be an all-zero frame with no real
    # content; exclude it so exactly one partial (tail) frame can exist.
    starts = [start for start in range(0, n_samples, hop_size) if start < n_samples]
    ends: list[int] = []
    for start in starts:
        ends.append(min(n_samples, start + frame_size))
    return len(starts), starts, ends


def extract_frames(wave: WaveData, config: PipelineConfig) -> FrameFeatures:
    frame_size = int(round(config.frame_length_ms * SAMPLE_RATE / 1000.0))
    hop_size = int(round(config.frame_shift_ms * SAMPLE_RATE / 1000.0))
    n_frames, starts, ends = frame_grid(wave.n_frames_pcm, frame_size, hop_size)

    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(frame_size) / max(1, frame_size - 1))
    filterbank = _mel_filterbank(config.mel_bands, frame_size, config.mel_fmax)
    dct_basis = _dct_basis(config.cepstrum_order, config.mel_bands)

    f0 = np.zeros(n_frames, dtype=np.float64)
    mel_energy = np.zeros((n_frames, config.mel_bands), dtype=np.float64)
    cepstrum = np.zeros((n_frames, config.cepstrum_order), dtype=np.float64)
    energy_db = np.zeros(n_frames, dtype=np.float64)
    is_tail = np.zeros(n_frames, dtype=bool)

    min_lag = max(1, int(np.floor(SAMPLE_RATE / _F0_MAX_HZ)))
    max_lag = min(frame_size - 1, int(np.ceil(SAMPLE_RATE / _F0_MIN_HZ)))

    for idx in range(n_frames):
        start, end = starts[idx], ends[idx]
        frame = np.zeros(frame_size, dtype=np.float64)
        frame[: end - start] = wave.samples[start:end]
        # Only the final frame is the designated zero-padded tail frame. Earlier
        # frames can be shorter than frame_size at the signal edge but are still
        # fully covered by valid samples and must not be mislabeled as tails.
        is_tail[idx] = (idx == n_frames - 1) and (end - start < frame_size)
        windowed = frame * window

        rms = float(np.sqrt(np.mean(frame * frame, dtype=np.float64)))
        energy_db[idx] = 10.0 * np.log10(rms * rms + _EPS)
        f0[idx] = _autocorr_f0(windowed, min_lag, max_lag)
        spectrum = np.fft.rfft(windowed).real ** 2 + np.fft.rfft(windowed).imag ** 2
        mel_e = filterbank @ spectrum
        log_mel = np.log(mel_e + _EPS)
        mel_energy[idx] = log_mel
        cepstrum[idx] = dct_basis @ log_mel

    sig = config.feature_sig_figs
    f0 = round_sigfigs(f0, sig)
    mel_energy = round_sigfigs(mel_energy, sig)
    cepstrum = round_sigfigs(cepstrum, sig)
    energy_db = round_sigfigs(energy_db, sig)

    feature_fp = sha256_text(
        fingerprint(
            {
                "sample_id": wave.sample_id,
                "n_pcm": wave.n_frames_pcm,
                "pcm_fp": wave.data_fp,
                "frame_size": frame_size,
                "hop_size": hop_size,
                "f0": f0,
                "mel_energy": mel_energy,
                "cepstrum": cepstrum,
                "energy_db": energy_db,
                "is_tail": [bool(x) for x in is_tail],
            }
        )
    )
    return FrameFeatures(
        sample_id=wave.sample_id,
        n_frames=n_frames,
        frame_size=frame_size,
        hop_size=hop_size,
        f0=f0,
        mel_energy=mel_energy,
        cepstrum=cepstrum,
        energy_db=energy_db,
        is_tail=is_tail,
        feature_fp=feature_fp,
    )


def _autocorr_f0(frame: np.ndarray, min_lag: int, max_lag: int) -> float:
    variance = float(np.dot(frame, frame))
    if variance <= _EPS:
        return 0.0
    # Unbiased autocorrelation over overlapping samples, deterministic argmax.
    corr = np.array(
        [np.dot(frame[: len(frame) - lag], frame[lag:]) for lag in range(min_lag, max_lag + 1)],
        dtype=np.float64,
    )
    peak_idx = int(np.argmax(corr))
    peak = float(corr[peak_idx])
    if peak <= 0.0:
        return 0.0
    lag = min_lag + peak_idx
    return SAMPLE_RATE / lag


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_bands: int, frame_size: int, fmax: float) -> np.ndarray:
    n_fft_bins = frame_size // 2 + 1
    fmin = 0.0
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(min(fmax, SAMPLE_RATE / 2.0)), n_bands + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mels], dtype=np.float64)
    bin_points = np.floor((frame_size + 1) * hz_points / SAMPLE_RATE).astype(np.int64)
    bank = np.zeros((n_bands, n_fft_bins), dtype=np.float64)
    for band in range(n_bands):
        left, center, right = bin_points[band], bin_points[band + 1], bin_points[band + 2]
        if right <= left:
            right = left + 2
            center = left + 1
        for bin_idx in range(left, min(right, n_fft_bins)):
            if bin_idx < center and center > left:
                bank[band, bin_idx] = (bin_idx - left) / (center - left)
            elif center <= bin_idx < right and right > center:
                bank[band, bin_idx] = (right - bin_idx) / (right - center)
    return bank


def _dct_basis(order: int, n_bands: int) -> np.ndarray:
    basis = np.zeros((order, n_bands), dtype=np.float64)
    for k in range(order):
        for n in range(n_bands):
            basis[k, n] = np.cos(np.pi * k * (n + 0.5) / n_bands)
        basis[k] *= np.sqrt(2.0 / n_bands)
    return basis
