"""Deterministic 16 kHz PCM synthesis, windowing, and fixed-point features.

The synthesizer never uses the ambient RNG.  Every random-looking value is
derived from the declared integer seed and a labelled SHA-256 stream, so
loading workers in another order cannot perturb the audio.
"""

from __future__ import annotations

import hashlib
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import canonical_bytes, sha256_hex
from .spec import WorkloadSpec


FEATURE_FRACTIONAL_BITS = 12
FEATURE_SCALE = 1 << FEATURE_FRACTIONAL_BITS
FEATURE_SCHEME = "spectral_band_amplitude_ratio_q12"


def _derived_bytes(*parts: Any, length: int = 32) -> bytes:
    return hashlib.sha256(canonical_bytes(list(parts))).digest()[:length]


def _derived_u64(*parts: Any) -> int:
    return int.from_bytes(_derived_bytes(*parts, length=8), "little")


@dataclass(frozen=True)
class SpeakerProfile:
    speaker_id: str
    f0_hz: int
    f0_contour_bps: int
    formants_hz: tuple[int, int, int]
    formant_bandwidths_hz: tuple[int, int, int]
    phase_offsets: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "f0_hz": self.f0_hz,
            "f0_contour_bps": self.f0_contour_bps,
            "formants_hz": list(self.formants_hz),
            "formant_bandwidths_hz": list(self.formant_bandwidths_hz),
        }


@dataclass(frozen=True)
class SpeechSegment:
    speaker_id: str
    start_sample: int
    end_sample: int
    overlap_with_previous: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "overlap_with_previous": self.overlap_with_previous,
        }


@dataclass(frozen=True)
class WindowTruth:
    window_id: str
    start_sample: int
    end_sample: int
    truth_speaker: str
    active_speakers: tuple[str, ...]
    coverage_bps: tuple[tuple[str, int], ...]
    overlap_bps: int
    truth_source: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "truth_speaker": self.truth_speaker,
            "active_speakers": list(self.active_speakers),
            "coverage_bps": {speaker: value for speaker, value in self.coverage_bps},
            "overlap_bps": self.overlap_bps,
            "truth_source": self.truth_source,
            "explanation": self.explanation,
        }


@dataclass
class SyntheticBundle:
    spec_digest: str
    pcm: np.ndarray
    speaker_profiles: tuple[SpeakerProfile, ...]
    segments: tuple[SpeechSegment, ...]
    windows: tuple[WindowTruth, ...]
    feature_dim: int
    features: np.ndarray
    feature_digest: str
    window_digest: str
    pcm_digest: str
    truth_digest: str

    @property
    def window_ids(self) -> tuple[str, ...]:
        return tuple(window.window_id for window in self.windows)

    def truth_for(self, window_id: str) -> WindowTruth:
        for window in self.windows:
            if window.window_id == window_id:
                return window
        raise KeyError(window_id)

    def feature_for(self, window_id: str, model_dim: int | None = None) -> tuple[int, ...]:
        index = self.window_ids.index(window_id)
        vector = self.features[index]
        if model_dim is None or model_dim == self.feature_dim:
            selected = vector
        elif model_dim <= 0:
            raise ValueError("model_dim must be positive")
        else:
            indices = np.linspace(0, self.feature_dim - 1, model_dim, dtype=np.int64)
            selected = vector[indices]
        return tuple(int(value) for value in selected)

    def truth_manifest(self) -> dict[str, Any]:
        return {
            "schema": "speaker-diag-window-truth-v1",
            "spec_digest": self.spec_digest,
            "window_digest": self.window_digest,
            "pcm_digest": self.pcm_digest,
            "truth_digest": self.truth_digest,
            "windows": [window.to_dict() for window in self.windows],
        }

    def feature_manifest(self) -> dict[str, Any]:
        records = []
        for index, window in enumerate(self.windows):
            packed = np.asarray(self.features[index], dtype="<i4").tobytes(order="C")
            records.append(
                {
                    "window_id": window.window_id,
                    "start_sample": window.start_sample,
                    "feature_digest": sha256_hex(packed),
                    "features": [int(value) for value in self.features[index]],
                }
            )
        return {
            "schema": "speaker-diag-feature-manifest-v1",
            "scheme": FEATURE_SCHEME,
            "fractional_bits": FEATURE_FRACTIONAL_BITS,
            "truncation": "floor",
            "feature_dim": self.feature_dim,
            "feature_digest": self.feature_digest,
            "windows": records,
        }


def speaker_profiles(spec: WorkloadSpec) -> tuple[SpeakerProfile, ...]:
    base_f0 = (88, 148, 208)
    base_formants = ((520, 1180, 2450), (690, 1620, 2730), (430, 980, 2240))
    profiles: list[SpeakerProfile] = []
    for index in range(spec.audio.speaker_count):
        digest = _derived_bytes("speaker", spec.seed, index, spec.audio.speaker_count, length=64)
        f0_delta = int.from_bytes(digest[0:2], "little") % 31
        contour_bps = 250 + int.from_bytes(digest[2:4], "little") % 900
        formants = tuple(
            base_formants[index][formant] + (int.from_bytes(digest[4 + formant * 2 : 6 + formant * 2], "little") % 90)
            for formant in range(3)
        )
        bandwidths = (
            70 + int.from_bytes(digest[10:12], "little") % 40,
            100 + int.from_bytes(digest[12:14], "little") % 60,
            150 + int.from_bytes(digest[14:16], "little") % 90,
        )
        phases = tuple(
            (int.from_bytes(digest[16 + offset : 18 + offset], "little") % 628318) / 100000.0
            for offset in (0, 2, 4, 6, 8, 10)
        )
        profiles.append(
            SpeakerProfile(
                speaker_id=f"spk_{index + 1}",
                f0_hz=base_f0[index] + f0_delta,
                f0_contour_bps=contour_bps,
                formants_hz=formants,
                formant_bandwidths_hz=bandwidths,
                phase_offsets=phases,
            )
        )
    return tuple(profiles)


def speech_segments(spec: WorkloadSpec, profiles: tuple[SpeakerProfile, ...]) -> tuple[SpeechSegment, ...]:
    sample_rate = spec.audio.sample_rate
    total_samples = spec.total_samples
    pause_min = spec.audio.pause_min_ms * sample_rate // 1000
    pause_max = spec.audio.pause_max_ms * sample_rate // 1000
    overlap_numerator = spec.audio.overlap_bps
    initial_gap = 20 * sample_rate // 1000
    cursor = initial_gap
    rotation = _derived_u64("rotation", spec.seed, spec.audio.speaker_count) % len(profiles)
    turn = 0
    segments: list[SpeechSegment] = []
    while cursor < total_samples:
        digest = _derived_bytes("turn", spec.seed, turn, len(profiles), length=32)
        duration_ms = 380 + int.from_bytes(digest[0:2], "little") % 440
        duration = max(1, duration_ms * sample_rate // 1000)
        make_overlap = bool(segments) and (int.from_bytes(digest[2:4], "little") % 10_000) < overlap_numerator
        previous_end = segments[-1].end_sample if segments else 0
        overlap_samples = 0
        if make_overlap:
            overlap_samples = int.from_bytes(digest[4:6], "little") % max(1, min(140 * sample_rate // 1000, previous_end))
        start = max(0, cursor - overlap_samples)
        end = min(total_samples, start + duration)
        if end <= start:
            break
        speaker_index = (rotation + turn) % len(profiles)
        segments.append(
            SpeechSegment(
                speaker_id=profiles[speaker_index].speaker_id,
                start_sample=start,
                end_sample=end,
                overlap_with_previous=make_overlap and start < previous_end,
            )
        )
        pause = pause_min
        if pause_max > pause_min:
            pause += int.from_bytes(digest[6:8], "little") % (pause_max - pause_min + 1)
        cursor = end + pause
        turn += 1
        if turn > 100_000:
            raise RuntimeError("speaker turn limit exceeded")
    if not segments:
        segments.append(SpeechSegment(profiles[0].speaker_id, 0, total_samples, False))
    return tuple(segments)


def _speaker_waveform(profile: SpeakerProfile, spec: WorkloadSpec) -> np.ndarray:
    sample_rate = spec.audio.sample_rate
    count = spec.total_samples
    t = np.arange(count, dtype=np.float64) / float(sample_rate)
    contour = (
        np.sin(2.0 * math.pi * 0.37 * t + profile.phase_offsets[0])
        + 0.45 * np.sin(2.0 * math.pi * 0.083 * t + profile.phase_offsets[1])
    )
    contour_ratio = profile.f0_contour_bps / 10_000.0
    f0 = profile.f0_hz * (1.0 + contour_ratio * contour)
    phase = 2.0 * math.pi * np.cumsum(f0, dtype=np.float64) / sample_rate
    waveform = np.zeros(count, dtype=np.float64)
    max_harmonic = max(1, min(32, int(4000 // profile.f0_hz)))
    for harmonic in range(1, max_harmonic + 1):
        frequency = profile.f0_hz * harmonic
        amplitude = 0.12 / math.sqrt(harmonic)
        for formant, bandwidth in zip(profile.formants_hz, profile.formant_bandwidths_hz):
            distance = (frequency - formant) / bandwidth
            amplitude += 0.85 / (1.0 + distance * distance) / math.sqrt(harmonic)
        waveform += amplitude * np.sin(
            harmonic * phase + profile.phase_offsets[(harmonic + 2) % len(profile.phase_offsets)]
        )
    peak = float(np.max(np.abs(waveform))) if count else 0.0
    if peak > 0.0:
        waveform /= peak
    return waveform


def _fade_gate(length: int, sample_rate: int) -> np.ndarray:
    gate = np.ones(length, dtype=np.float64)
    fade = min(length // 2, max(1, 8 * sample_rate // 1000))
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float64)
        gate[:fade] *= ramp
        gate[-fade:] *= ramp[::-1]
    return gate


def synthesize_pcm(spec: WorkloadSpec, profiles: tuple[SpeakerProfile, ...], segments: tuple[SpeechSegment, ...]) -> tuple[np.ndarray, np.ndarray]:
    count = spec.total_samples
    active = np.zeros((len(profiles), count), dtype=bool)
    gates = np.zeros((len(profiles), count), dtype=np.float64)
    speaker_index = {profile.speaker_id: index for index, profile in enumerate(profiles)}
    for segment in segments:
        index = speaker_index[segment.speaker_id]
        active[index, segment.start_sample : segment.end_sample] = True
        gates[index, segment.start_sample : segment.end_sample] = 1.0
        local = gates[index, segment.start_sample : segment.end_sample]
        local *= _fade_gate(segment.end_sample - segment.start_sample, spec.audio.sample_rate)
    mixed = np.zeros(count, dtype=np.float64)
    for index, profile in enumerate(profiles):
        if np.any(active[index]):
            mixed += gates[index] * _speaker_waveform(profile, spec)
    peak = float(np.max(np.abs(mixed))) if count else 0.0
    if peak > 0.0:
        mixed *= 0.86 / peak
    pcm = np.clip(np.rint(mixed * 32767.0), -32768, 32767).astype("<i2")
    return pcm, active


def _window_truths(spec: WorkloadSpec, profiles: tuple[SpeakerProfile, ...], active: np.ndarray) -> tuple[WindowTruth, ...]:
    truths: list[WindowTruth] = []
    speaker_ids = [profile.speaker_id for profile in profiles]
    for index, start in enumerate(spec.window_starts_samples):
        end = start + spec.window_samples
        block = active[:, start:end]
        counts = np.sum(block, axis=1, dtype=np.int64)
        active_indices = [speaker_index for speaker_index, count in enumerate(counts) if int(count) > 0]
        active_speakers = tuple(speaker_ids[speaker_index] for speaker_index in active_indices)
        coverage = tuple(
            (speaker_ids[speaker_index], int(counts[speaker_index]) * 10_000 // spec.window_samples)
            for speaker_index in range(len(profiles))
        )
        if active_indices:
            best_count = int(np.max(counts))
            best_indices = [speaker_index for speaker_index in active_indices if int(counts[speaker_index]) == best_count]
            truth_speaker = speaker_ids[min(best_indices)]
        else:
            truth_speaker = "silence"
        overlap_count = int(np.sum(np.sum(block, axis=0) >= 2))
        overlap_bps = overlap_count * 10_000 // spec.window_samples
        if not active_indices:
            truth_source = "silence"
        elif len(active_indices) > 1:
            truth_source = "overlap_dominant"
        else:
            truth_source = "single_active"
        explanation = (
            f"dominant={truth_speaker}; active={','.join(active_speakers) or 'none'}; "
            f"overlap={overlap_bps}bps; source={truth_source}"
        )
        truths.append(
            WindowTruth(
                window_id=f"W{index + 1:04d}",
                start_sample=start,
                end_sample=end,
                truth_speaker=truth_speaker,
                active_speakers=active_speakers,
                coverage_bps=coverage,
                overlap_bps=overlap_bps,
                truth_source=truth_source,
                explanation=explanation,
            )
        )
    return tuple(truths)


def extract_fixed_features(spec: WorkloadSpec, pcm: np.ndarray, feature_dim: int) -> np.ndarray:
    """Extract integer band-amplitude ratios with an explicit floor truncation.

    The returned shape is ``(window_count, feature_dim)`` and every cell is in
    ``[0, FEATURE_SCALE]``.  A zero-energy frame contributes a zero vector.
    """
    if feature_dim < 1:
        raise ValueError("feature_dim must be positive")
    frame_length = 256
    frame_hop = frame_length // 2
    window_length = spec.window_samples
    if window_length < frame_length:
        frame_length = max(16, 1 << int(math.floor(math.log2(window_length))))
        frame_hop = max(1, frame_length // 2)
    window = np.hanning(frame_length).astype(np.float64)
    bins = np.fft.rfftfreq(frame_length, d=1.0 / spec.audio.sample_rate)
    edges = np.searchsorted(bins, np.linspace(60.0, 3900.0, feature_dim + 1), side="left")
    edges[0] = 0
    edges[-1] = len(bins)
    frames_per_window = 1 + max(0, (window_length - frame_length) // frame_hop)
    result = np.zeros((len(spec.window_starts_samples), feature_dim), dtype=np.int64)
    signal = np.asarray(pcm, dtype=np.float64)
    for window_index, start in enumerate(spec.window_starts_samples):
        accum = np.zeros(feature_dim, dtype=np.float64)
        nonempty = 0
        for frame_index in range(frames_per_window):
            frame_start = start + frame_index * frame_hop
            frame = signal[frame_start : frame_start + frame_length]
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))
            spectrum = np.abs(np.fft.rfft(frame * window, n=frame_length))
            total = float(np.sum(spectrum))
            if total <= 0.0:
                continue
            nonempty += 1
            for band_index in range(feature_dim):
                left = int(edges[band_index])
                right = int(edges[band_index + 1])
                if right <= left:
                    band_amplitude = float(spectrum[min(left, len(spectrum) - 1)])
                else:
                    band_amplitude = float(np.sum(spectrum[left:right]))
                accum[band_index] += band_amplitude / total
        if nonempty:
            for band_index in range(feature_dim):
                result[window_index, band_index] = int(
                    math.floor(accum[band_index] * FEATURE_SCALE / nonempty)
                )
    return result


def synthesize(spec: WorkloadSpec, feature_dim: int | None = None) -> SyntheticBundle:
    if feature_dim is None:
        feature_dim = max(worker.model_dim for worker in spec.workers)
    profiles = speaker_profiles(spec)
    segments = speech_segments(spec, profiles)
    pcm, active = synthesize_pcm(spec, profiles, segments)
    windows = _window_truths(spec, profiles, active)
    features = extract_fixed_features(spec, pcm, feature_dim)
    pcm_bytes = pcm.astype("<i2", copy=False).tobytes(order="C")
    feature_bytes = features.astype("<i4", copy=False).tobytes(order="C")
    truth_obj = [window.to_dict() for window in windows]
    return SyntheticBundle(
        spec_digest=spec.spec_digest,
        pcm=pcm,
        speaker_profiles=profiles,
        segments=segments,
        windows=windows,
        feature_dim=feature_dim,
        features=features,
        feature_digest=sha256_hex(feature_bytes),
        window_digest=sha256_hex(canonical_bytes([window.start_sample for window in windows])),
        pcm_digest=sha256_hex(pcm_bytes),
        truth_digest=sha256_hex(canonical_bytes(truth_obj)),
    )


def write_wav(path: str | Path, bundle: SyntheticBundle, sample_rate: int = 16_000) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bundle.pcm.astype("<i2", copy=False).tobytes(order="C"))


def deterministic_audio_manifest(spec: WorkloadSpec, bundle: SyntheticBundle) -> dict[str, Any]:
    return {
        "schema": "speaker-diag-audio-manifest-v1",
        "spec_digest": spec.spec_digest,
        "seed": spec.seed,
        "duration_ms": spec.audio.duration_ms,
        "sample_rate": spec.audio.sample_rate,
        "channels": spec.audio.channels,
        "sample_width_bits": spec.audio.sample_width_bits,
        "pcm_bytes": int(bundle.pcm.nbytes),
        "pcm_digest": bundle.pcm_digest,
        "window_starts_sample": list(spec.window_starts_samples),
        "window_digest": bundle.window_digest,
        "truth_digest": bundle.truth_digest,
        "feature_scheme": FEATURE_SCHEME,
        "feature_fractional_bits": FEATURE_FRACTIONAL_BITS,
        "feature_truncation": "floor",
        "feature_dim": bundle.feature_dim,
        "feature_digest": bundle.feature_digest,
        "speakers": [profile.to_dict() for profile in bundle.speaker_profiles],
        "segments": [segment.to_dict() for segment in bundle.segments],
    }
