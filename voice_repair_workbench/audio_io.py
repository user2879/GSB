"""多格式素材解析与统一归一化。

解析结果统一为：
  raw_float: float64，形状 (samples, channels)，取值 [-1, 1]
  mono_analysis: 显式平均下混得到的分析用单声道 float64
并附带采样率、原始位深、声道数与有损格式标记。

声道策略：绝不静默取首通道。单声道与双声道被显式区分；调用方若声明了
期望声道数而与文件不符，抛 E_CHANNEL_MISMATCH。
"""
from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from . import config, lossy_codec
from .errors import E_CHANNEL_MISMATCH, E_FORMAT_UNSUPPORTED, RepairError
from .hashing import sha256_array, sha256_bytes


def _subtype_to_bit_depth(subtype: str | None) -> int | None:
    mapping = {
        "PCM_S8": 8,
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "DOUBLE": 64,
        "FLOAT": 32,
    }
    return mapping.get(subtype or "")


def downmix_to_mono(signal: np.ndarray, channels: int) -> np.ndarray:
    """显式等权平均下混；单声道直接返回拷贝。"""
    if channels == 1:
        return signal[:, 0].copy()
    if channels != 2:
        raise RepairError(
            E_FORMAT_UNSUPPORTED,
            f"only mono/stereo supported, got {channels} channels",
            field="channels",
            context={"channels": channels},
        )
    return np.mean(signal, axis=1).astype(np.float64)


class ParsedAudio:
    def __init__(
        self,
        path: str,
        raw_float: np.ndarray,
        sample_rate: int,
        bit_depth: int | None,
        channels: int,
        lossy: bool,
        lossy_format: str | None,
        file_hash: str,
    ) -> None:
        self.path = path
        self.raw_float = raw_float
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.channels = channels
        self.lossy = lossy
        self.lossy_format = lossy_format
        self.file_hash = file_hash
        self.mono_analysis = downmix_to_mono(raw_float, channels)

    def content_hash(self) -> str:
        """以解析后的原始浮点内容为内容哈希（缓存键的素材部分）。"""
        return sha256_array(self.raw_float)

    def to_meta(self) -> dict:
        return {
            "path": self.path,
            "sample_rate": self.sample_rate,
            "bit_depth": self.bit_depth,
            "channels": self.channels,
            "lossy": self.lossy,
            "lossy_format": self.lossy_format,
            "file_sha256": self.file_hash,
            "content_sha256": self.content_hash(),
            "frames": int(self.raw_float.shape[0]),
        }


def _read_file_bytes_hash(path: str) -> tuple[bytes, str]:
    with open(path, "rb") as fh:
        blob = fh.read()
    return blob, sha256_bytes(blob)


def parse_audio(path: str, expected_channels: int | None = None) -> ParsedAudio:
    if not os.path.exists(path):
        raise RepairError(
            E_FORMAT_UNSUPPORTED, f"audio file not found: {path}", field="path"
        )
    ext = os.path.splitext(path)[1].lower()

    if ext == config.LOSSY_EXTENSION:
        blob, file_hash = _read_file_bytes_hash(path)
        mono, sr = lossy_codec.decode(blob)
        raw = mono.reshape(-1, 1)
        parsed = ParsedAudio(
            path=path,
            raw_float=raw,
            sample_rate=sr,
            bit_depth=None,
            channels=1,
            lossy=True,
            lossy_format=config.LOSSY_FORMAT_NAME,
            file_hash=file_hash,
        )
    elif ext == ".wav":
        blob, file_hash = _read_file_bytes_hash(path)
        info = sf.info(path)
        if info.samplerate != config.SAMPLE_RATE:
            # 本工作台声明的统一入口为 48k；其他采样率显式拒绝而非隐式猜测。
            raise RepairError(
                E_FORMAT_UNSUPPORTED,
                f"only {config.SAMPLE_RATE} Hz WAV is supported, got {info.samplerate}",
                field="sample_rate",
                context={"sample_rate": info.samplerate},
            )
        bit_depth = _subtype_to_bit_depth(info.subtype)
        if bit_depth not in config.SUPPORTED_BIT_DEPTHS:
            raise RepairError(
                E_FORMAT_UNSUPPORTED,
                f"unsupported bit depth subtype: {info.subtype}",
                field="bit_depth",
                context={"subtype": info.subtype},
            )
        data, sr = sf.read(path, dtype="float64", always_2d=True)
        channels = data.shape[1]
        if channels not in config.SUPPORTED_CHANNELS:
            raise RepairError(
                E_FORMAT_UNSUPPORTED,
                f"only {config.SUPPORTED_CHANNELS} channels supported, got {channels}",
                field="channels",
            )
        parsed = ParsedAudio(
            path=path,
            raw_float=np.ascontiguousarray(data),
            sample_rate=sr,
            bit_depth=bit_depth,
            channels=channels,
            lossy=False,
            lossy_format=None,
            file_hash=file_hash,
        )
    else:
        raise RepairError(
            E_FORMAT_UNSUPPORTED,
            f"unsupported extension: {ext}",
            field="extension",
            context={"extension": ext},
        )

    if expected_channels is not None and parsed.channels != expected_channels:
        raise RepairError(
            E_CHANNEL_MISMATCH,
            (
                f"channel mismatch: expected {expected_channels} but file has "
                f"{parsed.channels} channels ({os.path.basename(path)})"
            ),
            field="channels",
            context={
                "expected": expected_channels,
                "actual": parsed.channels,
                "path": path,
            },
        )
    return parsed


def write_pcm_wav(path: str, mono_float: np.ndarray, sample_rate: int, bit_depth: int) -> str:
    """写出 PCM WAV（16/24bit）。量化采用确定性四舍五入，不做抖动。"""
    if bit_depth not in config.SUPPORTED_BIT_DEPTHS:
        raise RepairError(
            E_FORMAT_UNSUPPORTED,
            f"output bit depth must be 16 or 24, got {bit_depth}",
            field="bit_depth",
        )
    subtype = {16: "PCM_16", 24: "PCM_24"}[bit_depth]
    clipped = np.clip(mono_float, -1.0, 1.0)
    if bit_depth == 16:
        max_val = float(2**15 - 1)
        pcm = np.round(clipped * max_val).astype(np.int16)
    else:
        # numpy 无 int24：量化到 24bit 电平后回到 [-1,1] 浮点，由 soundfile 落 PCM_24
        max_val = float(2**23 - 1)
        pcm = (np.round(clipped * max_val) / max_val).astype(np.float64)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, pcm, sample_rate, subtype=subtype)
    return path
