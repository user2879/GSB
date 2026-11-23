"""WAV decoding, strict format validation and shard slicing."""

import io
import math
import os
import wave

import numpy as np

from . import fingerprints as fp
from .errors import E_AUDIO_FORMAT, PipelineError

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1


def list_wav_files(sample_dir):
    files = []
    for name in sorted(os.listdir(sample_dir)):
        if name.lower().endswith(".wav"):
            files.append(os.path.join(sample_dir, name))
    return files


def _read_raw(path):
    with open(path, "rb") as handle:
        return handle.read()


def validate_wav(path, injection=None):
    """Validate the RIFF container and PCM parameters.

    ``injection`` optionally mutates the byte stream before parsing so that
    injected failures are real parsing/validation failures rather than flags.
    """
    name = os.path.basename(path)
    raw = _read_raw(path)
    if injection and injection["type"] == "truncate_wav":
        keep_ratio = float(injection.get("keep_ratio", 0.6))
        raw = raw[: max(8, int(len(raw) * keep_ratio))]

    if len(raw) < 12 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise PipelineError(E_AUDIO_FORMAT, f"{name}: missing RIFF/WAVE container", stage="audio")

    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            params = wav.getparams()
            n_frames = wav.getnframes()
            frames = wav.readframes(n_frames)
    except (wave.Error, EOFError) as exc:
        raise PipelineError(E_AUDIO_FORMAT, f"{name}: corrupted/truncated WAV: {exc}", stage="audio")

    if params.nchannels != CHANNELS:
        raise PipelineError(E_AUDIO_FORMAT, f"{name}: expected mono, got {params.nchannels} channels",
                            stage="audio")
    if params.sampwidth != SAMPLE_WIDTH:
        raise PipelineError(E_AUDIO_FORMAT, f"{name}: expected 16bit, got {params.sampwidth * 8}bit",
                            stage="audio")
    if params.framerate != SAMPLE_RATE:
        raise PipelineError(E_AUDIO_FORMAT, f"{name}: expected 16000Hz, got {params.framerate}Hz",
                            stage="audio")

    expected_bytes = n_frames * params.nchannels * params.sampwidth
    if len(frames) != expected_bytes:
        raise PipelineError(
            E_AUDIO_FORMAT,
            f"{name}: data block length {len(frames)} != expected {expected_bytes}",
            stage="audio",
        )

    samples = np.frombuffer(frames, dtype="<i2").astype(np.float64)
    if not np.all(np.isfinite(samples)):
        raise PipelineError(E_AUDIO_FORMAT, f"{name}: non-finite samples detected", stage="audio")
    if injection and injection["type"] == "non_finite_samples":
        samples = samples.copy()
        target = int(injection.get("sample_index", 0)) % max(1, samples.shape[0])
        samples[target] = math.nan
        if not np.all(np.isfinite(samples)):
            raise PipelineError(E_AUDIO_FORMAT, f"{name}: non-finite samples detected",
                                stage="audio", injection_id=injection["id"])

    return {
        "audio": name,
        "path": path,
        "samples": samples,
        "n_samples": int(samples.shape[0]),
        "raw_fingerprint": fp.sha256_bytes(raw),
    }


def plan_shards(n_samples, shard_duration_ms, overlap_ms, frame_shift_ms):
    """Return shard sample spans [start, end) on a frame-shift aligned grid."""
    shard_len = SAMPLE_RATE * shard_duration_ms // 1000
    overlap_len = SAMPLE_RATE * overlap_ms // 1000
    step = shard_len - overlap_len
    shift = SAMPLE_RATE * frame_shift_ms // 1000
    # Align boundaries to the frame shift so frame indices are exact integers.
    step = max(shift, (step // shift) * shift)
    shard_len = (shard_len // shift) * shift
    shards = []
    start = 0
    index = 0
    while start < n_samples:
        end = min(n_samples, start + shard_len)
        shards.append({"shard_index": index, "sample_start": int(start), "sample_end": int(end)})
        if end >= n_samples:
            break
        start += step
        index += 1
    return shards
