"""WAV（标准库 wave）读写、RIFF 校验与片段/命令流加载校验层。"""

import json
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import BusinessError, E_AUDIO_FORMAT, E_FRAGMENT_INVALID, E_SAMPLE_NON_FINITE

TARGET_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2


@dataclass(frozen=True)
class Fragment:
    seq: int
    wav_file: str
    arrival_ms: int
    capture_ms: int
    command_word: str | None
    samples: np.ndarray  # int16，全局序号由 seq 决定
    sample_count: int


def parse_riff(raw: bytes) -> dict:
    """手工解析 RIFF/WAV 容器，返回参数与 data 块字节；不截断、不静默跳过。"""
    if len(raw) < 12 or raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise BusinessError(E_AUDIO_FORMAT, "缺少 RIFF/WAVE 容器头")
    offset = 12
    fmt = None
    data_blob = None
    declared_size = struct.unpack("<I", raw[4:8])[0]
    if declared_size + 8 != len(raw):
        raise BusinessError(
            E_AUDIO_FORMAT,
            f"RIFF chunk 尺寸 {declared_size} 与文件实际长度 {len(raw) - 8} 不一致",
        )
    while offset + 8 <= len(raw):
        chunk_id = raw[offset:offset + 4]
        chunk_size = struct.unpack("<I", raw[offset + 4:offset + 8])[0]
        body_start = offset + 8
        body_end = body_start + chunk_size
        if body_end > len(raw):
            raise BusinessError(
                E_AUDIO_FORMAT,
                f"{chunk_id.decode('ascii', 'replace')} 块声明 {chunk_size} 字节但文件被截断",
            )
        body = raw[body_start:body_end]
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise BusinessError(E_AUDIO_FORMAT, "fmt 块长度不足 16 字节")
            audio_format, channels, rate, _byte_rate, _block_align, bits = struct.unpack(
                "<HHIIHH", body[:16]
            )
            fmt = {
                "audio_format": audio_format,
                "channels": channels,
                "rate": rate,
                "bits": bits,
            }
        elif chunk_id == b"data":
            data_blob = body
        offset = body_end + (chunk_size & 1)
    if fmt is None:
        raise BusinessError(E_AUDIO_FORMAT, "缺少 fmt 块")
    if data_blob is None:
        raise BusinessError(E_AUDIO_FORMAT, "缺少 data 块")
    fmt["data_bytes"] = data_blob
    return fmt


def _validate_params(fmt: dict) -> None:
    if fmt["audio_format"] != 1:
        raise BusinessError(E_AUDIO_FORMAT, f"仅支持 PCM(format=1)，实际 {fmt['audio_format']}")
    if fmt["rate"] != TARGET_RATE:
        raise BusinessError(E_AUDIO_FORMAT, f"采样率必须为 {TARGET_RATE}，实际 {fmt['rate']}")
    if fmt["channels"] != TARGET_CHANNELS:
        raise BusinessError(E_AUDIO_FORMAT, f"必须单声道，实际 {fmt['channels']} 声道")
    if fmt["bits"] != 16:
        raise BusinessError(E_AUDIO_FORMAT, f"位深必须为 16bit，实际 {fmt['bits']}bit")


def decode_pcm(fmt: dict) -> np.ndarray:
    data_blob = fmt["data_bytes"]
    expected_bytes = len(data_blob)
    if expected_bytes % 2 != 0:
        raise BusinessError(E_AUDIO_FORMAT, "data 块长度不是 16bit 采样的整数倍")
    samples = np.frombuffer(data_blob, dtype="<i2").astype(np.int16)
    if samples.nbytes != expected_bytes:
        raise BusinessError(E_AUDIO_FORMAT, "data 块长度与采样数不一致")
    if not np.all(np.isfinite(samples.astype(np.float64))):
        raise BusinessError(E_SAMPLE_NON_FINITE, "存在 NaN/Inf 采样值")
    return samples


def _check_finite(samples: np.ndarray) -> None:
    view = samples.astype(np.float64)
    if not np.all(np.isfinite(view)):
        raise BusinessError(E_SAMPLE_NON_FINITE, "存在 NaN/Inf 采样值")


def read_wav_bytes(raw: bytes, source: str = "<bytes>",
                   inject_non_finite: bool = False) -> np.ndarray:
    """对 WAV 字节做完整双通道校验，支持注入后的字节流。"""
    fmt = parse_riff(raw)
    _validate_params(fmt)
    import io
    with wave.open(io.BytesIO(raw), "rb") as handle:
        if handle.getframerate() != TARGET_RATE:
            raise BusinessError(E_AUDIO_FORMAT, "wave 模块读到的采样率不一致")
        if handle.getnchannels() != TARGET_CHANNELS:
            raise BusinessError(E_AUDIO_FORMAT, "wave 模块读到的声道数不一致")
        if handle.getsampwidth() != TARGET_SAMPLE_WIDTH:
            raise BusinessError(E_AUDIO_FORMAT, "wave 模块读到的位深不一致")
        wave_bytes = handle.readframes(handle.getnframes())
        if len(wave_bytes) != len(fmt["data_bytes"]):
            raise BusinessError(E_AUDIO_FORMAT, "data 块长度和 wave 模块读到的帧数不一致")
        frame_count = handle.getnframes()
    samples = decode_pcm(fmt)
    if inject_non_finite:
        buffer = samples.astype(np.float64)
        buffer[0] = float("nan")
        _check_finite(buffer)
    _check_finite(samples)
    if len(samples) != frame_count:
        raise BusinessError(
            E_AUDIO_FORMAT,
            f"data 块采样数 {len(samples)} 与帧数 {frame_count} 不一致",
        )
    return samples


def read_wav(path: str | Path) -> np.ndarray:
    """标准库 wave 与手工 RIFF 双通道校验，返回 int16 一维数组。"""
    wav_path = Path(path)
    if not wav_path.exists():
        raise BusinessError(E_AUDIO_FORMAT, f"WAV 文件不存在: {wav_path}")
    return read_wav_bytes(wav_path.read_bytes(), str(wav_path))


def write_wav(path: str | Path, samples: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.asarray(samples, dtype="<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(TARGET_CHANNELS)
        handle.setsampwidth(TARGET_SAMPLE_WIDTH)
        handle.setframerate(TARGET_RATE)
        handle.writeframes(pcm)


def _validate_manifest_entries(entries: list[dict], vocab_words: set[str]) -> None:
    seqs = []
    for entry in entries:
        seq = entry.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
            raise BusinessError(
                E_FRAGMENT_INVALID, f"片段序号必须为非负整数: {entry.get('seq')!r}"
            )
        seqs.append(seq)
        for key in ("arrival_ms", "capture_ms"):
            value = entry.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise BusinessError(
                    E_FRAGMENT_INVALID,
                    f"片段 {seq} 的 {key} 必须为非负整数，实际 {value!r}",
                    fragment_seq=seq,
                )
        word = entry.get("command_word")
        if word is not None and word not in vocab_words:
            raise BusinessError(
                E_FRAGMENT_INVALID,
                f"片段 {seq} 命令词 {word!r} 不在受控词表内",
                fragment_seq=seq,
            )
        if not isinstance(entry.get("wav_file"), str) or not entry["wav_file"]:
            raise BusinessError(
                E_FRAGMENT_INVALID, f"片段 {seq} 缺少 wav_file", fragment_seq=seq
            )
    if len(set(seqs)) != len(seqs):
        dupes = sorted({s for s in seqs if seqs.count(s) > 1})
        raise BusinessError(E_FRAGMENT_INVALID, f"片段序号重复: {dupes}")


def load_manifest(manifest_path: str | Path, wav_root: str | Path,
                  vocab_words: set[str]) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("fragments")
    if not isinstance(entries, list) or not entries:
        raise BusinessError(E_FRAGMENT_INVALID, "manifest 缺少非空 fragments 列表")
    _validate_manifest_entries(entries, vocab_words)
    root = Path(wav_root)
    resolved = []
    for entry in sorted(entries, key=lambda item: item["seq"]):
        wav_path = (root / entry["wav_file"]).resolve()
        resolved.append({**entry, "wav_path": str(wav_path)})
    return resolved


def load_fragments(manifest_path: str | Path, wav_root: str | Path,
                   vocab_words: set[str]) -> list[Fragment]:
    entries = load_manifest(manifest_path, wav_root, vocab_words)
    fragments = []
    for entry in entries:
        samples = read_wav(entry["wav_path"])
        fragments.append(
            Fragment(
                seq=entry["seq"],
                wav_file=entry["wav_file"],
                arrival_ms=entry["arrival_ms"],
                capture_ms=entry["capture_ms"],
                command_word=entry.get("command_word"),
                samples=samples,
                sample_count=len(samples),
            )
        )
    return fragments


def load_command_stream(path: str | Path, vocab_words: set[str]) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    stream = payload.get("commands")
    if not isinstance(stream, list):
        raise BusinessError(E_FRAGMENT_INVALID, "命令流缺少 commands 列表")
    for index, item in enumerate(stream):
        if not isinstance(item.get("fragment_seq"), int) or isinstance(
            item["fragment_seq"], bool
        ):
            raise BusinessError(E_FRAGMENT_INVALID, f"命令流第 {index} 项 fragment_seq 非法")
        word = item.get("word")
        if word not in vocab_words:
            raise BusinessError(
                E_FRAGMENT_INVALID,
                f"命令流第 {index} 项命令词 {word!r} 不在受控词表内",
                fragment_seq=item["fragment_seq"],
            )
        if not isinstance(item.get("issued_ms"), int) or item["issued_ms"] < 0:
            raise BusinessError(E_FRAGMENT_INVALID, f"命令流第 {index} 项 issued_ms 非法")
    return stream
