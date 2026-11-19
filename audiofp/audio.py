"""Strict local audio decoding, provenance capture and format normalization."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from .errors import AudioFormatError, EncoderError

try:  # soundfile is part of the declared runtime, but keep the error actionable.
    import soundfile as sf
except Exception:  # pragma: no cover - exercised on deliberately bare hosts
    sf = None


@dataclass(frozen=True)
class SourceFormat:
    container: str
    codec: str
    sample_rate: int
    bit_depth: int | None
    channels: int
    declared_frames: int
    decoded_frames: int
    duration: float
    file_size: int
    sha256: str
    path: str

    @property
    def format_signature(self) -> str:
        depth = "lossy" if self.bit_depth is None else str(self.bit_depth)
        return f"{self.container.lower()}:{self.codec.lower()}:{self.sample_rate}:{depth}:{self.channels}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "container": self.container,
            "codec": self.codec,
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "declared_frames": self.declared_frames,
            "decoded_frames": self.decoded_frames,
            "duration": round(self.duration, 9),
            "file_size": self.file_size,
            "sha256": self.sha256,
            "format_signature": self.format_signature,
        }


@dataclass
class AudioDocument:
    samples: np.ndarray
    analysis_rate: int
    source: SourceFormat
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.analysis_rate)


@dataclass(frozen=True)
class _WavHeader:
    channels: int
    sample_rate: int
    block_align: int
    bit_depth: int
    declared_frames: int
    actual_frames: int
    audio_format: int


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_ffmpeg() -> str | None:
    candidates: list[str] = []
    configured = os.environ.get("AUDIOFP_FFMPEG")
    if configured:
        candidates.append(configured)
    located = shutil.which("ffmpeg")
    if located:
        candidates.append(located)
    try:  # A local wheel-provided binary keeps the runtime fully offline.
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    return None


def _probe_wav_header(path: Path) -> _WavHeader:
    file_size = path.stat().st_size
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
            if len(header) != 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise ValueError("not a RIFF/WAVE file")
            declared_riff_size = struct.unpack("<I", header[4:8])[0]
            if declared_riff_size + 8 > file_size:
                raise ValueError(
                    f"RIFF size {declared_riff_size} exceeds file size {file_size}"
                )

            fmt: tuple[int, int, int, int, int, int] | None = None
            declared_data_size: int | None = None
            actual_data_bytes: int | None = None
            while handle.tell() + 8 <= file_size:
                chunk_id = handle.read(4)
                size_bytes = handle.read(4)
                if len(size_bytes) != 4:
                    raise ValueError("truncated chunk header")
                chunk_size = struct.unpack("<I", size_bytes)[0]
                data_start = handle.tell()
                if data_start + chunk_size > file_size:
                    if chunk_id == b"data":
                        actual_data_bytes = max(0, file_size - data_start)
                        declared_data_size = chunk_size
                        break
                    raise ValueError(f"truncated {chunk_id!r} chunk")
                if chunk_id == b"fmt ":
                    if chunk_size < 16:
                        raise ValueError("fmt chunk is too short")
                    values = struct.unpack("<HHIIHH", handle.read(16))
                    audio_format, channels, sample_rate, byte_rate, block_align, bits = values
                    if audio_format == 0xFFFE and chunk_size >= 40:
                        extension = handle.read(chunk_size - 16)
                        if len(extension) >= 24:
                            valid_bits = struct.unpack("<H", extension[0:2])[0]
                            subformat = extension[8:24]
                            if subformat[:2] == b"\x01\x00":
                                audio_format = 1
                            if valid_bits:
                                bits = valid_bits
                    fmt = (audio_format, channels, sample_rate, byte_rate, block_align, bits)
                    handle.seek(data_start + chunk_size + (chunk_size & 1))
                elif chunk_id == b"data":
                    declared_data_size = chunk_size
                    actual_data_bytes = chunk_size
                    break
                else:
                    handle.seek(data_start + chunk_size + (chunk_size & 1))

            if fmt is None:
                raise ValueError("missing fmt chunk")
            if declared_data_size is None or actual_data_bytes is None:
                raise ValueError("missing data chunk")
            audio_format, channels, sample_rate, byte_rate, block_align, bits = fmt
            if audio_format not in (1, 3):
                raise ValueError(f"unsupported WAV format tag {audio_format}")
            if channels <= 0 or sample_rate <= 0 or bits <= 0 or block_align <= 0:
                raise ValueError("invalid fmt chunk values")
            expected_block_align = channels * math.ceil(bits / 8)
            if block_align != expected_block_align:
                raise ValueError(
                    f"block_align {block_align} != channels*bits/8 ({expected_block_align})"
                )
            if byte_rate != sample_rate * block_align:
                raise ValueError(
                    f"byte_rate {byte_rate} != sample_rate*block_align ({sample_rate * block_align})"
                )
            if actual_data_bytes < declared_data_size:
                raise ValueError(
                    f"declared data bytes {declared_data_size} exceed available {actual_data_bytes}"
                )
            if declared_data_size % block_align:
                raise ValueError("data chunk size is not frame aligned")
            if actual_data_bytes < declared_data_size:
                raise ValueError("truncated audio data")
            return _WavHeader(
                channels=channels,
                sample_rate=sample_rate,
                block_align=block_align,
                bit_depth=bits,
                declared_frames=declared_data_size // block_align,
                actual_frames=actual_data_bytes // block_align,
                audio_format=audio_format,
            )
    except AudioFormatError:
        raise
    except Exception as exc:
        raise AudioFormatError(
            f"Invalid or truncated WAV header: {path.name}",
            {"path": path.name, "reason": str(exc)},
        ) from exc


def _soundfile_bit_depth(subtype: str | None) -> int | None:
    if not subtype:
        return None
    match = re.search(r"PCM_(?:S|U)?(8|16|24|32)", subtype, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if subtype.upper() in {"FLOAT", "DOUBLE"}:
        return 32 if subtype.upper() == "FLOAT" else 64
    # VORBIS/OPUS/MPEG are intrinsically lossy and have no source bit depth.
    return None


def _channel_count(layout: str) -> int:
    normalized = layout.strip().lower()
    if normalized in {"mono", "1 channels", "1 channel"}:
        return 1
    if normalized in {"stereo", "2 channels", "2 channel"}:
        return 2
    match = re.search(r"(\d+)\s*channels?", normalized)
    if match:
        return int(match.group(1))
    if normalized.startswith("5.1"):
        return 6
    if normalized.startswith("7.1"):
        return 8
    return 0


def _ffmpeg_probe(path: Path, ffmpeg: str | None = None) -> dict[str, Any]:
    binary = ffmpeg or find_ffmpeg()
    if not binary:
        raise AudioFormatError(
            "soundfile cannot decode this file and no local ffmpeg was found",
            {"path": path.name},
        )
    process = subprocess.run(
        [binary, "-hide_banner", "-i", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    text = process.stderr
    match = re.search(
        r"Audio:\s*([^,\n]+),\s*(\d+)\s*Hz,\s*([^,\n]+)",
        text,
        re.IGNORECASE,
    )
    if not match:
        safe_stderr = text[-500:].replace(str(path.resolve()), path.name).replace(
            str(path), path.name
        )
        raise AudioFormatError(
            "ffmpeg could not identify an audio stream",
            {"path": path.name, "stderr_tail": safe_stderr},
        )
    codec, rate_text, layout = match.groups()
    return {
        "codec": codec.strip().upper(),
        "sample_rate": int(rate_text),
        "channels": _channel_count(layout) or 1,
        "layout": layout.strip(),
    }


def _decode_with_ffmpeg(
    path: Path,
    target_rate: int,
    ffmpeg: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    binary = ffmpeg or find_ffmpeg()
    if not binary:
        raise AudioFormatError(
            "No local decoder is available for this container",
            {"path": path.name, "container": path.suffix.lower()},
        )
    probe = _ffmpeg_probe(path, binary)
    process = subprocess.run(
        [
            binary,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(target_rate),
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0 or not process.stdout:
        safe_stderr = process.stderr.decode("utf-8", "replace")[-500:]
        safe_stderr = safe_stderr.replace(str(path.resolve()), path.name).replace(
            str(path), path.name
        )
        raise AudioFormatError(
            "ffmpeg decode failed",
            {
                "path": path.name,
                "returncode": process.returncode,
                "stderr_tail": safe_stderr,
            },
        )
    if len(process.stdout) % 4:
        raise AudioFormatError(
            "ffmpeg returned a non-frame-aligned PCM stream",
            {"path": path.name, "bytes": len(process.stdout)},
        )
    samples = np.frombuffer(process.stdout, dtype="<f4").astype(np.float64)
    probe["decoded_at_analysis_rate"] = target_rate
    return samples, probe


def _compare_expected(source: SourceFormat, expected: dict[str, Any] | None) -> None:
    if not expected:
        return
    aliases = {
        "samplerate": "sample_rate",
        "sample_rate": "sample_rate",
        "channels": "channels",
        "bit_depth": "bit_depth",
        "bits": "bit_depth",
        "container": "container",
        "codec": "codec",
        "frames": "declared_frames",
        "declared_frames": "declared_frames",
        "decoded_frames": "decoded_frames",
    }
    mismatches: list[dict[str, Any]] = []
    source_dict = source.to_dict()
    for key, value in expected.items():
        canonical = aliases.get(str(key).lower())
        if not canonical or value is None:
            continue
        actual = source_dict.get(canonical)
        if canonical in {"container", "codec"} and isinstance(actual, str):
            equal = actual.upper() == str(value).upper()
        else:
            equal = actual == value
        if not equal:
            mismatches.append({"field": canonical, "declared": value, "actual": actual})
    if mismatches:
        raise AudioFormatError(
            "Declared audio parameters do not match decoded audio",
            {"path": source.path, "mismatches": mismatches},
        )


def decode_audio(
    path: str | os.PathLike[str],
    target_rate: int,
    expected: dict[str, Any] | None = None,
    ffmpeg: str | None = None,
) -> AudioDocument:
    """Decode one local file, validate provenance and normalize to mono/float64.

    There is no fallback to guessed sample rates.  A corrupt header, unsupported
    codec, short data chunk or declared/actual frame mismatch raises
    ``E_AUDIO_FORMAT``.
    """

    audio_path = Path(path)
    if not audio_path.is_file():
        raise AudioFormatError("Audio file does not exist", {"path": str(path)})
    if audio_path.stat().st_size == 0:
        raise AudioFormatError("Audio file is empty", {"path": audio_path.name})

    suffix = audio_path.suffix.lower()
    warnings: list[str] = []
    header: _WavHeader | None = None
    if suffix == ".wav":
        header = _probe_wav_header(audio_path)

    data: np.ndarray | None = None
    source_rate: int
    channels: int
    bit_depth: int | None
    container: str
    codec: str
    declared_frames: int

    if sf is not None:
        try:
            info = sf.info(str(audio_path))
            decoded, decoded_rate = sf.read(
                str(audio_path),
                dtype="float64",
                always_2d=True,
            )
            if decoded.ndim != 2 or decoded.shape[0] == 0:
                raise ValueError("decoder returned no frames")
            source_rate = int(info.samplerate)
            channels = int(info.channels)
            if int(decoded_rate) != source_rate:
                raise ValueError("decoder sample rate differs from file metadata")
            if decoded.shape[1] != channels:
                raise ValueError("decoder channel count differs from file metadata")
            if int(info.frames) != decoded.shape[0]:
                raise ValueError(
                    f"metadata declares {info.frames} frames but decoder returned {decoded.shape[0]}"
                )
            if header is not None:
                if header.channels != channels or header.sample_rate != source_rate:
                    raise ValueError("soundfile metadata differs from strict WAV header")
                if header.declared_frames != decoded.shape[0]:
                    raise ValueError(
                        f"WAV declares {header.declared_frames} frames but {decoded.shape[0]} decoded"
                    )
            data = decoded
            container = str(info.format or suffix.lstrip(".")).upper()
            codec = str(info.subtype or "UNKNOWN").upper()
            bit_depth = _soundfile_bit_depth(info.subtype)
            declared_frames = int(info.frames)
        except Exception as exc:
            if header is not None:
                raise AudioFormatError(
                    "WAV decoding failed; declared parameters were not accepted as defaults",
                    {
                        "path": audio_path.name,
                        "declared": {
                            "sample_rate": header.sample_rate,
                            "channels": header.channels,
                            "frames": header.declared_frames,
                            "bit_depth": header.bit_depth,
                        },
                        "reason": str(exc),
                    },
                ) from exc
            warnings.append(f"soundfile decode failed: {exc}")

    if data is None:
        decoded, probe = _decode_with_ffmpeg(audio_path, target_rate, ffmpeg)
        sample_count_at_source = len(decoded) * probe["sample_rate"] / target_rate
        return _finalize_document(
            mono=decoded,
            already_at_target=True,
            target_rate=target_rate,
            source=SourceFormat(
                container=suffix.lstrip(".").upper() or "UNKNOWN",
                codec=probe["codec"],
                sample_rate=int(probe["sample_rate"]),
                bit_depth=None,
                channels=int(probe["channels"]),
                declared_frames=int(round(sample_count_at_source)),
                decoded_frames=int(round(sample_count_at_source)),
                duration=sample_count_at_source / float(probe["sample_rate"]),
                file_size=audio_path.stat().st_size,
                sha256=sha256_file(audio_path),
                path=audio_path.name,
            ),
            expected=expected,
            warnings=warnings,
        )

    assert data is not None
    if not np.isfinite(data).all():
        raise AudioFormatError(
            "Decoded samples contain NaN or infinity",
            {"path": audio_path.name},
        )
    if channels <= 0 or source_rate <= 0 or data.shape[0] <= 0:
        raise AudioFormatError(
            "Decoded audio has invalid dimensions",
            {
                "path": audio_path.name,
                "sample_rate": source_rate,
                "channels": channels,
                "frames": int(data.shape[0]),
            },
        )

    mono = data.mean(axis=1, dtype=np.float64)
    if source_rate != target_rate:
        divisor = math.gcd(source_rate, target_rate)
        up = target_rate // divisor
        down = source_rate // divisor
        expected_length = int(round(len(mono) * target_rate / source_rate))
        mono = resample_poly(mono, up, down, window=("kaiser", 5.0))
        # resample_poly already returns the intended length; this keeps the
        # roundoff contract explicit and stable across SciPy point releases.
        if len(mono) != expected_length and expected_length > 0:
            if len(mono) > expected_length:
                mono = mono[:expected_length]
            else:
                mono = np.pad(mono, (0, expected_length - len(mono)))

    source = SourceFormat(
        container=container,
        codec=codec,
        sample_rate=source_rate,
        bit_depth=bit_depth,
        channels=channels,
        declared_frames=declared_frames,
        decoded_frames=int(data.shape[0]),
        duration=data.shape[0] / float(source_rate),
        file_size=audio_path.stat().st_size,
        sha256=sha256_file(audio_path),
        path=audio_path.name,
    )
    return _finalize_document(
        mono=mono,
        already_at_target=True,
        target_rate=target_rate,
        source=source,
        expected=expected,
        warnings=warnings,
    )


def _finalize_document(
    mono: np.ndarray,
    already_at_target: bool,
    target_rate: int,
    source: SourceFormat,
    expected: dict[str, Any] | None,
    warnings: list[str],
) -> AudioDocument:
    if not already_at_target:  # Defensive: callers should normalize first.
        divisor = math.gcd(source.sample_rate, target_rate)
        mono = resample_poly(
            mono,
            target_rate // divisor,
            source.sample_rate // divisor,
            window=("kaiser", 5.0),
        )
    if len(mono) == 0:
        raise AudioFormatError(
            "Decoded audio contains zero frames after normalization",
            {"path": source.path, "sample_rate": source.sample_rate},
        )
    if not np.isfinite(mono).all():
        raise AudioFormatError(
            "Normalized audio contains non-finite samples",
            {"path": source.path},
        )
    _compare_expected(source, expected)
    return AudioDocument(
        samples=np.asarray(mono, dtype=np.float64),
        analysis_rate=target_rate,
        source=source,
        warnings=warnings,
    )


def write_wav(
    path: str | os.PathLike[str],
    samples: np.ndarray,
    sample_rate: int,
    *,
    subtype: str = "PCM_16",
    channels: int | None = None,
) -> None:
    """Write a local WAV with soundfile, validating the requested shape."""

    if sf is None:
        raise AudioFormatError("soundfile is required to write WAV fixtures")
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        array_in = array
        actual_channels = 1
    else:
        actual_channels = int(array.shape[1])
        array_in = array
    if channels is not None and channels != actual_channels:
        raise ValueError(f"requested {channels} channels but got {actual_channels}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path),
        np.clip(array_in, -1.0, 1.0),
        int(sample_rate),
        format="WAV",
        subtype=subtype,
    )


def resample_to(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float64).copy()
    divisor = math.gcd(int(source_rate), int(target_rate))
    result = resample_poly(
        np.asarray(samples, dtype=np.float64),
        int(target_rate) // divisor,
        int(source_rate) // divisor,
        window=("kaiser", 5.0),
    )
    expected = int(round(len(samples) * target_rate / source_rate))
    if len(result) > expected:
        result = result[:expected]
    elif len(result) < expected:
        result = np.pad(result, (0, expected - len(result)))
    return np.asarray(result, dtype=np.float64)


def encode_lossy_ogg(
    source_wav: str | os.PathLike[str],
    output_ogg: str | os.PathLike[str],
    *,
    quality: int = 5,
    ffmpeg: str | None = None,
) -> None:
    """Encode with the user-provided, local ffmpeg binary. No network is used."""

    binary = ffmpeg or find_ffmpeg()
    if not binary:
        raise EncoderError(
            "No local ffmpeg binary is configured; set AUDIOFP_FFMPEG",
            {"source": str(source_wav)},
        )
    Path(output_ogg).parent.mkdir(parents=True, exist_ok=True)
    command = [
        binary,
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(source_wav),
        "-c:a",
        "libvorbis",
        "-q:a",
        str(quality),
        "-map_metadata",
        "-1",
        str(output_ogg),
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode != 0 or not Path(output_ogg).is_file():
        raise EncoderError(
            "Local lossy encoding failed",
            {
                "command": command,
                "returncode": process.returncode,
                "stderr_tail": process.stderr.decode("utf-8", "replace")[-1000:],
            },
        )
    normalize_ogg_stream_serial(output_ogg)


def _ogg_crc_table() -> tuple[int, ...]:
    table: list[int] = []
    for value in range(256):
        register = value << 24
        for _ in range(8):
            if register & 0x80000000:
                register = ((register << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                register = (register << 1) & 0xFFFFFFFF
        table.append(register)
    return tuple(table)


_OGG_CRC_TABLE = _ogg_crc_table()


def _ogg_crc(payload: bytes) -> int:
    register = 0
    for value in payload:
        register = ((register << 8) & 0xFFFFFFFF) ^ _OGG_CRC_TABLE[
            ((register >> 24) ^ value) & 0xFF
        ]
    return register


def normalize_ogg_stream_serial(
    path: str | os.PathLike[str],
    serial: int = 0x41554446,
) -> None:
    """Make Ogg/Vorbis bytes reproducible without changing its audio stream.

    The Ogg muxer chooses a random bitstream serial and therefore random page
    checksums on every encode.  Rewriting the serial to a fixed value and
    recomputing the Ogg CRC keeps the Vorbis packets, decoded samples and
    format semantics unchanged while making the local artifact byte-stable.
    """

    ogg_path = Path(path)
    payload = bytearray(ogg_path.read_bytes())
    position = 0
    pages = 0
    while position < len(payload):
        if payload[position : position + 4] != b"OggS":
            raise EncoderError(
                "Encoded file is not a valid Ogg stream",
                {"path": ogg_path.name, "offset": position},
            )
        if position + 27 > len(payload):
            raise EncoderError("Truncated Ogg page header", {"path": ogg_path.name})
        segment_count = payload[position + 26]
        header_size = 27 + segment_count
        if position + header_size > len(payload):
            raise EncoderError("Truncated Ogg segment table", {"path": ogg_path.name})
        body_size = sum(payload[position + 27 : position + header_size])
        page_size = header_size + body_size
        if position + page_size > len(payload):
            raise EncoderError("Truncated Ogg page body", {"path": ogg_path.name})
        struct.pack_into("<I", payload, position + 14, serial)
        payload[position + 22 : position + 26] = b"\x00\x00\x00\x00"
        checksum = _ogg_crc(bytes(payload[position : position + page_size]))
        struct.pack_into("<I", payload, position + 22, checksum)
        position += page_size
        pages += 1
    if pages == 0:
        raise EncoderError("Ogg encoder produced no pages", {"path": ogg_path.name})
    ogg_path.write_bytes(payload)
