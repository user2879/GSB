from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .biz import DATA_CHUNK_MISMATCH, WAV_HEADER_INVALID, WAV_FILE_MISSING, issue_dict


SUPPORTED_BIT_DEPTHS = (8, 16, 24, 32)


def _uint32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def parse_wav_container(path: str | Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    path = Path(path)
    issues: list[dict[str, Any]] = []
    if not path.exists():
        issues.append(issue_dict(WAV_FILE_MISSING, str(path), "WAV file does not exist"))
        return None, issues

    raw = path.read_bytes()
    if len(raw) < 44 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        issues.append(issue_dict(WAV_HEADER_INVALID, str(path), "missing RIFF/WAVE header"))
        return None, issues

    riff_size = _uint32(raw, 4)
    if riff_size + 8 != len(raw):
        issues.append(
            issue_dict(
                WAV_HEADER_INVALID,
                str(path),
                "RIFF size does not match file length",
                riff_size=riff_size,
                file_size=len(raw),
            )
        )
        return None, issues

    pos = 12
    fmt: dict[str, Any] | None = None
    data_offset: int | None = None
    data_size: int | None = None
    chunks: list[dict[str, int | str]] = []

    while pos + 8 <= len(raw):
        chunk_id = raw[pos : pos + 4]
        try:
            chunk_name = chunk_id.decode("ascii")
        except UnicodeDecodeError:
            chunk_name = "????"
        size = _uint32(raw, pos + 4)
        body_offset = pos + 8
        chunks.append({"id": chunk_name, "offset": pos, "size": size})
        if body_offset + size > len(raw):
            issues.append(
                issue_dict(
                    DATA_CHUNK_MISMATCH,
                    str(path),
                    f"chunk {chunk_name} extends past end of file",
                    chunk_size=size,
                    body_offset=body_offset,
                    file_size=len(raw),
                )
            )
            return None, issues

        if chunk_id == b"fmt " and size >= 16:
            audio_format, channels, sample_rate, byte_rate, block_align, bits = struct.unpack_from(
                "<HHIIHH", raw, body_offset
            )
            fmt = {
                "audio_format": audio_format,
                "channels": channels,
                "sample_rate": sample_rate,
                "byte_rate": byte_rate,
                "block_align": block_align,
                "bit_depth": bits,
            }
        elif chunk_id == b"data":
            data_offset = body_offset
            data_size = size

        pos = body_offset + size + (size & 1)

    if fmt is None or data_offset is None or data_size is None:
        issues.append(issue_dict(WAV_HEADER_INVALID, str(path), "missing fmt or data chunk", chunks=chunks))
        return None, issues

    if fmt["audio_format"] != 1:
        issues.append(
            issue_dict(WAV_HEADER_INVALID, str(path), "only PCM WAV is supported", audio_format=fmt["audio_format"])
        )
        return None, issues
    if fmt["channels"] != 1:
        issues.append(issue_dict(WAV_HEADER_INVALID, str(path), "only mono clips are generated and accepted"))
        return None, issues
    if fmt["bit_depth"] not in SUPPORTED_BIT_DEPTHS:
        issues.append(
            issue_dict(
                WAV_HEADER_INVALID,
                str(path),
                "unsupported bit depth",
                bit_depth=fmt["bit_depth"],
                supported=list(SUPPORTED_BIT_DEPTHS),
            )
        )
        return None, issues

    bytes_per_sample = fmt["bit_depth"] // 8
    expected_block_align = bytes_per_sample * fmt["channels"]
    if fmt["block_align"] != expected_block_align:
        issues.append(
            issue_dict(
                WAV_HEADER_INVALID,
                str(path),
                "block alignment is inconsistent",
                block_align=fmt["block_align"],
                expected=expected_block_align,
            )
        )
        return None, issues

    sample_count = data_size // expected_block_align
    if data_size != sample_count * expected_block_align:
        issues.append(
            issue_dict(
                DATA_CHUNK_MISMATCH,
                str(path),
                "data chunk length is not an integral sample count",
                data_size=data_size,
                block_align=expected_block_align,
            )
        )
        return None, issues

    info = {
        "path": str(path),
        "format": "pcm_wav",
        "channels": fmt["channels"],
        "sample_rate": fmt["sample_rate"],
        "bit_depth": fmt["bit_depth"],
        "bytes_per_sample": bytes_per_sample,
        "block_align": expected_block_align,
        "data_offset": data_offset,
        "data_size": data_size,
        "sample_count": sample_count,
        "riff_size": riff_size,
        "file_size": len(raw),
        "chunks": chunks,
    }
    return info, issues


def encode_levels(levels: np.ndarray, bit_depth: int) -> bytes:
    levels = np.asarray(levels, dtype=np.int64)
    if bit_depth == 8:
        return (levels + 128).astype(np.uint8).tobytes()
    if bit_depth == 16:
        return (levels * 256).astype("<i2").tobytes()
    if bit_depth == 32:
        return (levels * (1 << 24)).astype("<i4").tobytes()
    if bit_depth == 24:
        values = (levels * (1 << 16)).astype(np.int32)
        out = bytearray()
        for value in values.tolist():
            if value < 0:
                value += 1 << 24
            out.extend(value.to_bytes(3, byteorder="little", signed=False))
        return bytes(out)
    raise ValueError(f"unsupported bit depth: {bit_depth}")


def write_wav_from_levels(path: str | Path, levels: np.ndarray, sample_rate: int, bit_depth: int) -> dict[str, Any]:
    if bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise ValueError(f"unsupported bit depth: {bit_depth}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = encode_levels(levels, bit_depth)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(bit_depth // 8)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)
    info, issues = parse_wav_container(path)
    if issues or info is None:
        raise RuntimeError(f"generated invalid WAV: {issues}")
    return info


def _decode_samples(raw: bytes, bit_depth: int) -> np.ndarray:
    if bit_depth == 8:
        unsigned = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
        level = unsigned.astype(np.float64) - 128.0
        return level / 128.0
    width = bit_depth // 8
    if bit_depth == 16:
        integers = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        return integers / 32768.0
    if bit_depth == 32:
        integers = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        return integers / 2147483648.0
    if bit_depth == 24:
        padded = bytearray()
        for index in range(0, len(raw), 3):
            triple = raw[index : index + 3]
            value = int.from_bytes(triple, "little", signed=True)
            padded.extend(struct.pack("<i", value))
        integers = np.frombuffer(bytes(padded), dtype="<i4").astype(np.float64)
        return integers / 8388608.0
    raise ValueError(f"unsupported bit depth: {bit_depth}")


def read_wav_samples(path: str | Path) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    info, issues = parse_wav_container(path)
    if issues or info is None:
        return info or {}, np.zeros(0, dtype=np.float64), issues
    with wave.open(str(path), "rb") as wav:
        raw = wav.readframes(wav.getnframes())
    expected_bytes = info["sample_count"] * info["block_align"]
    if len(raw) != expected_bytes:
        issues.append(
            issue_dict(
                DATA_CHUNK_MISMATCH,
                str(path),
                "decoded frame byte length differs from data chunk",
                read_bytes=len(raw),
                expected_bytes=expected_bytes,
            )
        )
        return info, np.zeros(0, dtype=np.float64), issues
    samples = _decode_samples(raw, info["bit_depth"])
    return info, samples, issues
