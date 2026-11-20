"""Audio decoding and deterministic cross-correlation alignment."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from .constants import CHANNELS, FPS, SAMPLE_RATE, TIMEBASE
from .errors import BusinessError
from .util import hash_object


def audio_dependency_hash(
    source_audio_sha256: str,
    shots: list[dict],
    injection: dict,
) -> str:
    return hash_object(
        {
            "source_audio_sha256": source_audio_sha256,
            "shots": shots,
            "injection": {
                key: injection.get(key)
                for key in (
                    "audio_declared_sample_rate",
                    "audio_sample_offset",
                    "audio_gain",
                    "audio_tolerance_ms",
                    "timestamp_jumps",
                )
            },
        }
    )


def decode_wav(path: Path, declared_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except (OSError, wave.Error) as exc:
        raise BusinessError("E_DECODE", f"cannot decode WAV {path.name}", {"path": str(path)}) from exc
    if channels != CHANNELS or sample_width != 2:
        raise BusinessError(
            "E_DECODE",
            "expected stereo 16-bit PCM",
            {"channels": channels, "sample_width": sample_width},
        )
    if declared_sample_rate is not None and int(declared_sample_rate) != sample_rate:
        raise BusinessError(
            "E_AUDIO_SAMPLE_RATE",
            "declared audio sample rate differs from WAV header",
            {"declared": int(declared_sample_rate), "actual": sample_rate},
        )
    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, CHANNELS)
    return pcm.astype(np.float64) / 32768.0, sample_rate


def load_reference(path: Path) -> np.ndarray:
    try:
        data = np.fromfile(path, dtype="<f4")
    except OSError as exc:
        raise BusinessError("E_DECODE", f"cannot read reference audio {path.name}", {"path": str(path)}) from exc
    if data.size % CHANNELS:
        raise BusinessError("E_DECODE", "reference PCM has a partial stereo sample", {"samples": int(data.size)})
    return data.reshape(-1, CHANNELS).astype(np.float64)


def shift_audio(samples: np.ndarray, offset_samples: int) -> np.ndarray:
    if offset_samples == 0:
        return samples.copy()
    shifted = np.zeros_like(samples)
    if offset_samples > 0:
        if offset_samples < samples.shape[0]:
            shifted[offset_samples:] = samples[: samples.shape[0] - offset_samples]
    else:
        count = min(samples.shape[0], -offset_samples)
        shifted[: samples.shape[0] - count] = samples[count:]
    return shifted


def _estimate_lag(reference: np.ndarray, decoded: np.ndarray, max_lag: int) -> tuple[int, float, float]:
    if reference.size == 0 or decoded.size == 0:
        return 0, 1.0, 0.0
    ref = reference - float(reference.mean())
    dec = decoded - float(decoded.mean())
    nfft = 1 << max(1, (reference.size + decoded.size - 1).bit_length())
    cross = np.fft.irfft(np.fft.rfft(dec, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    positive = cross[: max_lag + 1]
    negative = cross[-max_lag:] if max_lag else np.empty(0, dtype=cross.dtype)
    candidates = np.concatenate((negative, positive))
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    lag = int(lags[int(np.argmax(candidates))])
    if lag >= 0:
        decoded_part = dec[lag:]
        reference_part = ref[: decoded_part.size]
    else:
        reference_part = ref[-lag:]
        decoded_part = dec[: reference_part.size]
    denominator = float(np.sqrt(np.dot(decoded_part, decoded_part) * np.dot(reference_part, reference_part)))
    similarity = float(np.dot(decoded_part, reference_part) / denominator) if denominator > 0.0 else 0.0
    reference_energy = float(np.dot(reference_part, reference_part))
    gain = float(np.dot(decoded_part, reference_part) / reference_energy) if reference_energy > 0.0 else 0.0
    return lag, gain, similarity


def _segment_video_offset_ms(shots: list[dict], timestamp_jumps: dict[str, Any]) -> list[float]:
    offsets: list[float] = []
    for shot in shots:
        start = int(shot["start_frame"])
        end = int(shot["end_frame"])
        values = [float(timestamp_jumps.get(str(frame), 0.0)) for frame in range(start, end + 1)]
        offsets.append(float(np.mean(values)) if values else 0.0)
    return offsets


def evaluate_audio(
    wav_path: Path,
    reference_path: Path,
    shots: list[dict],
    injection: dict,
    source_audio_sha256: str,
) -> dict:
    declared_rate = injection.get("audio_declared_sample_rate")
    try:
        decoded, actual_rate = decode_wav(wav_path, declared_rate)
        reference = load_reference(reference_path)
    except BusinessError as exc:
        result = {
            "schema_version": "audio-alignment/1.0",
            "timebase": TIMEBASE,
            "tolerance_ms": float(injection.get("audio_tolerance_ms", 1.0)),
            "segments": [],
            "overall": {
                "status": "error",
                "first_drift_segment": None,
                "max_abs_offset_ms": None,
                "joint_av_skew_ms": None,
            },
            "error": exc.to_dict(),
            "source_audio_sha256": source_audio_sha256,
            "injection_hash": hash_object(injection),
        }
        result["record_hash"] = hash_object(result)
        return result
    if reference.shape != decoded.shape:
        exc = BusinessError(
            "E_AUDIO_DRIFT",
            "reference and decoded audio lengths differ",
            {"decoded_samples": int(decoded.shape[0]), "reference_samples": int(reference.shape[0])},
        )
        result = {
            "schema_version": "audio-alignment/1.0",
            "timebase": TIMEBASE,
            "tolerance_ms": float(injection.get("audio_tolerance_ms", 1.0)),
            "segments": [],
            "overall": {"status": "error", "first_drift_segment": None, "max_abs_offset_ms": None, "joint_av_skew_ms": None},
            "error": exc.to_dict(),
            "source_audio_sha256": source_audio_sha256,
            "injection_hash": hash_object(injection),
        }
        result["record_hash"] = hash_object(result)
        return result
    decoded = shift_audio(decoded, int(injection.get("audio_sample_offset", 0)))
    decoded = decoded * float(injection.get("audio_gain", 1.0))
    reference_mono = reference.mean(axis=1)
    decoded_mono = decoded.mean(axis=1)
    max_lag = max(1, int(round(actual_rate * 0.05)))
    video_offsets = _segment_video_offset_ms(shots, injection.get("timestamp_jumps", {}))
    segments: list[dict] = []
    tolerance = float(injection.get("audio_tolerance_ms", 1.0))
    first_drift: int | None = None
    for index, shot in enumerate(shots):
        start_frame = int(shot["start_frame"])
        end_frame = int(shot["end_frame"])
        start_sample = int(round(start_frame * actual_rate / FPS))
        end_sample = int(round((end_frame + 1) * actual_rate / FPS))
        lag, gain, similarity = _estimate_lag(
            reference_mono[start_sample:end_sample],
            decoded_mono[start_sample:end_sample],
            max_lag,
        )
        offset_ms = lag * 1000.0 / actual_rate
        video_offset_ms = video_offsets[index]
        joint_skew_ms = offset_ms - video_offset_ms
        within = abs(offset_ms) <= tolerance
        if not within and first_drift is None:
            first_drift = int(shot["shot_id"])
        segments.append(
            {
                "shot_id": int(shot["shot_id"]),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "offset_samples": lag,
                "offset_ms": round(offset_ms, 9),
                "gain_ratio": round(gain, 9),
                "waveform_similarity": round(similarity, 9),
                "video_offset_ms": round(video_offset_ms, 9),
                "joint_av_skew_ms": round(joint_skew_ms, 9),
                "within_tolerance": within,
            }
        )
    overall_audio = float(np.mean([segment["offset_ms"] for segment in segments])) if segments else 0.0
    overall_video = float(np.mean(video_offsets)) if video_offsets else 0.0
    max_abs = max((abs(segment["offset_ms"]) for segment in segments), default=0.0)
    error = None
    if first_drift is not None:
        error = BusinessError(
            "E_AUDIO_DRIFT",
            "audio offset exceeds declared tolerance",
            {"first_drift_segment": first_drift, "tolerance_ms": tolerance, "max_abs_offset_ms": max_abs},
        ).to_dict()
    result = {
        "schema_version": "audio-alignment/1.0",
        "timebase": TIMEBASE,
        "sample_rate": actual_rate,
        "channels": CHANNELS,
        "reference_sample_count": int(reference.shape[0]),
        "tolerance_ms": tolerance,
        "segments": segments,
        "overall": {
            "status": "drift" if first_drift is not None else "ok",
            "first_drift_segment": first_drift,
            "max_abs_offset_ms": round(max_abs, 9),
            "audio_offset_ms": round(overall_audio, 9),
            "video_offset_ms": round(overall_video, 9),
            "joint_av_skew_ms": round(overall_audio - overall_video, 9),
        },
        "error": error,
        "source_audio_sha256": source_audio_sha256,
        "injection_hash": hash_object(injection),
        "deps_hash": audio_dependency_hash(source_audio_sha256, shots, injection),
    }
    result["record_hash"] = hash_object({key: value for key, value in result.items() if key != "record_hash"})
    return result
