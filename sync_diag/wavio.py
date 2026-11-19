from __future__ import annotations

import wave
import struct
from pathlib import Path
from typing import Iterator

import numpy as np

from .errors import require


PCM24_FULL_SCALE = 8_388_607.0


def validate_pcm24_wav(
    path: Path,
    expected_channels: int,
    expected_rate: int,
    expected_frames: int | None = None,
) -> dict[str, int]:
    try:
        with wave.open(str(path), "rb") as wav:
            info = {
                "channels": wav.getnchannels(),
                "bit_depth": wav.getsampwidth() * 8,
                "sample_rate_hz": wav.getframerate(),
                "frames": wav.getnframes(),
            }
    except (OSError, EOFError, wave.Error) as exc:
        require(False, "E_CH_FORMAT", "cannot read PCM WAV", path=str(path), reason=str(exc))
    require(
        info["channels"] == expected_channels,
        "E_CH_FORMAT",
        "channel count mismatch",
        path=str(path),
        expected=expected_channels,
        actual=info["channels"],
    )
    require(
        info["bit_depth"] == 24,
        "E_CH_FORMAT",
        "bit depth mismatch",
        path=str(path),
        expected=24,
        actual=info["bit_depth"],
    )
    require(
        info["sample_rate_hz"] == expected_rate,
        "E_CH_FORMAT",
        "sample rate mismatch",
        path=str(path),
        expected=expected_rate,
        actual=info["sample_rate_hz"],
    )
    if expected_frames is not None:
        require(
            info["frames"] == expected_frames,
            "E_CH_FORMAT",
            "frame count mismatch",
            path=str(path),
            expected=expected_frames,
            actual=info["frames"],
        )
    return info


def decode_pcm24(raw: bytes, channels: int) -> np.ndarray:
    bytes_u8 = np.frombuffer(raw, dtype=np.uint8)
    require(
        bytes_u8.size % (channels * 3) == 0,
        "E_CH_FORMAT",
        "PCM byte count is not frame aligned",
        bytes=bytes_u8.size,
        channels=channels,
    )
    frames = bytes_u8.size // (channels * 3)
    triples = bytes_u8.reshape(frames, channels, 3).astype(np.int32)
    values = triples[..., 0] | (triples[..., 1] << 8) | (triples[..., 2] << 16)
    values = np.where((values & 0x800000) != 0, values - 0x1000000, values)
    return (values.astype(np.float64) / PCM24_FULL_SCALE).astype(np.float32)


def encode_pcm24(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    values = np.rint(clipped * PCM24_FULL_SCALE).astype("<i4")
    raw = values.view(np.uint8).reshape(values.shape[0], values.shape[1] * 4)
    return np.ascontiguousarray(raw.reshape(values.shape[0], values.shape[1], 4)[:, :, :3]).tobytes()


def iter_wav_blocks(path: Path, block_frames: int) -> Iterator[tuple[int, np.ndarray]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        require(wav.getsampwidth() == 3, "E_CH_FORMAT", "only PCM24 is accepted", path=str(path))
        position = 0
        while True:
            raw = wav.readframes(block_frames)
            if not raw:
                break
            yield position, decode_pcm24(raw, channels)
            position += len(raw) // (channels * 3)


def write_wav_header(path: Path, channels: int, rate: int, frames: int) -> wave.Wave_write:
    handle = wave.open(str(path), "wb")
    handle.setnchannels(channels)
    handle.setsampwidth(3)
    handle.setframerate(rate)
    handle.setnframes(frames)
    return handle


def wav_header_bytes(channels: int, rate: int, frames: int) -> bytes:
    block_align = channels * 3
    data_size = frames * block_align
    header = bytearray(b"RIFF")
    header += struct.pack("<I", 36 + data_size)
    header += b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, channels, rate, rate * block_align, block_align, 24)
    header += b"data"
    header += struct.pack("<I", data_size)
    return bytes(header)
