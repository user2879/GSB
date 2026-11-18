"""Deterministic synthetic WAV generator with controllable speaker timbre."""
from __future__ import annotations

import os
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH
from .util import read_json, stable_seed

_DEFAULT_NOISE_FLOOR = 0.002
_DEFAULT_DURATION_S = 1.0


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_id: str
    f0_mean_hz: float
    f0_std_hz: float
    formants_hz: tuple[float, float, float]
    formant_gains: tuple[float, float, float]
    envelope_hz: float


def _speaker_profiles(spec: dict[str, Any]) -> SpeakerProfile:
    return SpeakerProfile(
        speaker_id=str(spec["speaker_id"]),
        f0_mean_hz=float(spec["f0_mean_hz"]),
        f0_std_hz=float(spec["f0_std_hz"]),
        formants_hz=tuple(float(x) for x in spec["formants_hz"]),
        formant_gains=tuple(float(x) for x in spec.get("formant_gains", (1.0, 0.7, 0.4))),
        envelope_hz=float(spec.get("envelope_hz", 4.0)),
    )


def _formant_filter(formants_hz: tuple[float, float, float],
                    gains: tuple[float, float, float], n_taps: int) -> np.ndarray:
    """Sum of Gaussian-shaped spectral resonances converted to an FIR kernel via irfft."""
    freqs = np.fft.rfftfreq(n_taps, d=1.0 / SAMPLE_RATE)
    response = np.ones_like(freqs) * 0.05
    for center, gain in zip(formants_hz, gains):
        bandwidth = max(40.0, center * 0.08)
        response += gain * np.exp(-0.5 * ((freqs - center) / bandwidth) ** 2)
    kernel = np.fft.irfft(response, n=n_taps)
    kernel = np.roll(kernel, n_taps // 2)
    kernel /= float(np.sum(np.abs(kernel)))
    return kernel


def synth_utterance(profile: SpeakerProfile, utterance_id: str,
                    duration_s: float = _DEFAULT_DURATION_S,
                    noise_floor: float = _DEFAULT_NOISE_FLOOR) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(f"{profile.speaker_id}|{utterance_id}"))
    n_samples = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE

    # Smooth F0 contour: deterministic mean with controlled frame-wise variance.
    block = max(1, SAMPLE_RATE // 100)  # 10 ms control grid
    n_blocks = int(np.ceil(n_samples / block))
    contour_blocks = profile.f0_mean_hz + profile.f0_std_hz * rng.standard_normal(n_blocks)
    contour_blocks = np.clip(contour_blocks, 50.0, 500.0)
    f0_contour = np.repeat(contour_blocks, block)[:n_samples]
    phase = 2.0 * np.pi * np.cumsum(f0_contour) / SAMPLE_RATE

    source = np.zeros(n_samples, dtype=np.float64)
    for harmonic in range(1, 9):
        source += (1.0 / harmonic) * np.sin(harmonic * phase + rng.uniform(0, 2 * np.pi))
    source /= float(np.max(np.abs(source)) + 1e-12)

    kernel = _formant_filter(profile.formants_hz, profile.formant_gains, 257)
    colored = np.convolve(source, kernel, mode="same")

    envelope = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(
        2.0 * np.pi * profile.envelope_hz * t + rng.uniform(0, 2 * np.pi)
    ))
    colored *= envelope
    noise = noise_floor * rng.standard_normal(n_samples)
    signal = colored + noise
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = 0.9 * signal / peak
    return signal.astype(np.float64)


def write_wav(path: str, samples: np.ndarray) -> None:
    pcm = np.clip(np.round(samples * 32768.0), -32768, 32767).astype("<i2")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def generate_dataset(input_dir: str, injection_path: str) -> dict[str, Any]:
    injection = read_json(injection_path)
    profiles = {str(s["speaker_id"]): _speaker_profiles(s)
                for s in injection["speaker_profiles"]}
    written: list[str] = []
    for entry in injection["utterances"]:
        profile = profiles[str(entry["speaker_id"])]
        utt_id = str(entry["utterance_id"])
        duration = float(entry.get("duration_s", _DEFAULT_DURATION_S))
        noise = float(entry.get("noise_floor", _DEFAULT_NOISE_FLOOR))
        samples = synth_utterance(profile, utt_id, duration, noise)
        path = os.path.join(input_dir, f"{utt_id}.wav")
        write_wav(path, samples)
        written.append(path)

    for bad in injection.get("invalid_audio", []):
        path = os.path.join(input_dir, str(bad["file_name"]))
        _write_invalid(path, str(bad["kind"]))
        written.append(path)
    return {"written": sorted(written), "count": len(written)}


def _write_invalid(path: str, kind: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if kind == "bad_rate":
        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(8000)
            wf.writeframes(np.zeros(1600, dtype="<i2").tobytes())
    elif kind == "stereo":
        with wave.open(path, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(np.zeros((800, 2), dtype="<i2").tobytes())
    elif kind == "bad_width":
        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(1)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(np.zeros(1600, dtype=np.uint8).tobytes())
    elif kind == "truncated":
        good = os.path.join(os.path.dirname(path), "_tmp_good.wav")
        with wave.open(good, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(np.zeros(1600, dtype="<i2").tobytes())
        with open(good, "rb") as handle:
            blob = bytearray(handle.read())
        os.remove(good)
        # Corrupt the data-size header so data length disagrees with the container header.
        blob[40:44] = (len(blob) - 44 + 99).to_bytes(4, "little")
        with open(path, "wb") as handle:
            handle.write(blob)
    else:
        raise ValueError(f"unknown invalid audio kind: {kind}")
