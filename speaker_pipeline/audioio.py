"""WAV container validation and PCM IO using the standard library ``wave``.

Only 16 kHz mono 16-bit PCM WAV is accepted.  Every validation failure
raises :class:`AudioFormatError` with business code ``E_AUDIO_FORMAT``.
"""

import os
import wave

import numpy as np

from .errors import AudioFormatError, NonFiniteSampleError

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
N_CHANNELS = 1


def validate_wav_container(path):
    """Validate the WAV header/container; raise on any rule violation."""
    if not os.path.exists(path):
        raise AudioFormatError(f"wav file not found: {path}", stage="io")
    try:
        with wave.open(path, "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            compression = wav.getcomptype()
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioFormatError(
            f"corrupt or unreadable wav container: {path}: {exc}",
            stage="io",
        ) from exc
    if compression != "NONE":
        raise AudioFormatError(
            f"unsupported compression {compression} in {path}", stage="io"
        )
    if channels != N_CHANNELS:
        raise AudioFormatError(
            f"expected mono audio, got {channels} channels: {path}",
            stage="io",
        )
    if sample_width != SAMPLE_WIDTH:
        raise AudioFormatError(
            f"expected 16-bit audio, got width {sample_width}: {path}",
            stage="io",
        )
    if sample_rate != SAMPLE_RATE:
        raise AudioFormatError(
            f"expected {SAMPLE_RATE} Hz, got {sample_rate}: {path}",
            stage="io",
        )
    return {
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": sample_rate,
        "n_frames": wav.getnframes() if False else _frame_count(path),
    }


def _frame_count(path):
    with wave.open(path, "rb") as wav:
        return wav.getnframes()


def read_wav(path):
    """Return ``(samples_float64, meta)`` for a validated WAV file."""
    meta = validate_wav_container(path)
    with wave.open(path, "rb") as wav:
        raw = wav.readframes(wav.getnframes())
    if len(raw) < meta["n_frames"] * SAMPLE_WIDTH:
        raise AudioFormatError(
            f"truncated PCM payload in {path}", stage="io"
        )
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return samples, meta


def assert_finite(samples, file_name, injection_id=None):
    """Reject NaN/Inf sample values (defensive; float64 synthetic data)."""
    if not np.all(np.isfinite(samples)):
        raise NonFiniteSampleError(
            f"non-finite sample values detected in {file_name}",
            stage="feature",
            injection_id=injection_id,
        )


def write_wav(path, samples):
    """Write float64 samples in [-1, 1] as 16 kHz mono 16-bit PCM."""
    samples = np.asarray(samples, dtype=np.float64)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(N_CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def list_wav_files(directory):
    """Deterministic sorted listing of ``.wav`` files (by file name)."""
    names = [
        name
        for name in os.listdir(directory)
        if name.lower().endswith(".wav")
    ]
    return [os.path.join(directory, name) for name in sorted(names)]
