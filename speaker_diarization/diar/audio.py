"""WAV 读取与 RIFF 容器严格校验。

校验项：RIFF/WAVE 容器头、PCM 格式、16kHz、单声道、16bit、
数据块长度与采样数一致；截断文件单独映射为 E_AUDIO_TRUNCATED。
"""

import numpy as np

from .codes import E_AUDIO_FORMAT, E_AUDIO_TRUNCATED, PipelineError
from .synth import SAMPLE_RATE


def _read_chunks(data):
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise PipelineError(E_AUDIO_FORMAT, "audio", "缺少 RIFF/WAVE 容器头")
    offset = 12
    chunks = {}
    while offset + 8 <= len(data):
        chunk_id = data[offset:offset + 4]
        chunk_size = int.from_bytes(data[offset + 4:offset + 8], "little")
        body_start = offset + 8
        body_end = body_start + chunk_size
        if body_end > len(data):
            chunks[chunk_id] = (body_start, len(data), chunk_size, True)
            break
        chunks[chunk_id] = (body_start, body_end, chunk_size, False)
        offset = body_end + (chunk_size & 1)  # 偶数字节填充
    return chunks


def read_wav_bytes(data, audio_name="<memory>"):
    chunks = _read_chunks(data)
    if b"fmt " not in chunks:
        raise PipelineError(E_AUDIO_FORMAT, "audio", "缺少 fmt 块")
    fmt_start, fmt_end, _, _ = chunks[b"fmt "]
    fmt = data[fmt_start:fmt_end]
    if len(fmt) < 16:
        raise PipelineError(E_AUDIO_FORMAT, "audio", "fmt 块长度不足")
    audio_format = int.from_bytes(fmt[0:2], "little")
    channels = int.from_bytes(fmt[2:4], "little")
    sample_rate = int.from_bytes(fmt[4:8], "little")
    bits = int.from_bytes(fmt[14:16], "little")
    if audio_format != 1:
        raise PipelineError(E_AUDIO_FORMAT, "audio", f"非 PCM 格式 tag={audio_format}")
    if channels != 1:
        raise PipelineError(E_AUDIO_FORMAT, "audio", f"非单声道 channels={channels}")
    if sample_rate != SAMPLE_RATE:
        raise PipelineError(E_AUDIO_FORMAT, "audio", f"采样率非 16000: {sample_rate}")
    if bits != 16:
        raise PipelineError(E_AUDIO_FORMAT, "audio", f"位深非 16bit: {bits}")
    if b"data" not in chunks:
        raise PipelineError(E_AUDIO_FORMAT, "audio", "缺少 data 块")
    data_start, data_end, declared_size, truncated = chunks[b"data"]
    available = data_end - data_start
    if truncated or available < declared_size:
        raise PipelineError(
            E_AUDIO_TRUNCATED, "audio",
            f"data 块声明 {declared_size} 字节，实际仅 {available} 字节",
        )
    if declared_size % 2 != 0:
        raise PipelineError(E_AUDIO_FORMAT, "audio", "data 块长度不是偶数")
    pcm = np.frombuffer(data[data_start:data_start + declared_size], dtype="<i2")
    return {
        "audio": audio_name,
        "samples": pcm.astype(np.float64) / 32768.0,
        "n_samples": int(pcm.shape[0]),
        "sample_rate": sample_rate,
        "channels": channels,
        "bits": bits,
        "data_bytes": declared_size,
    }


def read_wav(path):
    with open(path, "rb") as fh:
        return read_wav_bytes(fh.read(), audio_name=path.split("/")[-1].split("\\")[-1])
