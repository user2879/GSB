"""本地确定性语音素材合成器。

全部素材由固定种子的 numpy 随机数 + 确定性公式生成，不依赖真实录音、
外部素材或网络。同一配置两次生成的 WAV 文件逐字节一致。

时间线（每段固定长度）：
  对白(对话音调和谐波) -> 静音 -> 房间底噪(带色噪声) -> 对白 -> 静音 -> 底噪
"""
from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from . import config
from .errors import E_FORMAT_UNSUPPORTED, E_PARAM_RANGE, RepairError

MASTER_SEED = 20240517

# 段定义：(类型, 秒数)
SEGMENT_PLAN = (
    ("dialogue", 1.0),
    ("silence", 0.5),
    ("room_noise", 0.8),
    ("dialogue", 1.0),
    ("silence", 0.4),
    ("room_noise", 0.7),
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _dialogue_segment(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """模拟对白：基频缓慢漂移的谐波叠加 + 幅度包络（音节起伏）。"""
    t = np.arange(n, dtype=np.float64) / sr
    base_f0 = 120.0
    f0 = base_f0 + 18.0 * np.sin(2 * np.pi * 1.7 * t) + 6.0 * np.sin(2 * np.pi * 4.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    signal = np.zeros(n, dtype=np.float64)
    for harmonic, gain in ((1, 0.60), (2, 0.28), (3, 0.14), (4, 0.07), (5, 0.03)):
        signal += gain * np.sin(harmonic * phase + rng.uniform(0, 2 * np.pi))
    # 音节级幅度包络
    syllable = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t + 0.7))
    # 轻微确定性抖动（同一 rng 序列 -> 逐样本可复现）
    jitter = 1.0 + 0.02 * rng.standard_normal(n)
    return signal * syllable * jitter


def _room_noise_segment(n: int, rng: np.random.Generator) -> np.ndarray:
    """房间底噪：白噪声经一阶低通得到有色噪声，幅度较低。"""
    white = rng.standard_normal(n)
    filtered = np.empty(n, dtype=np.float64)
    alpha = 0.18
    acc = 0.0
    for i in range(n):  # 确定性 IIR，跨平台一致
        acc = alpha * white[i] + (1.0 - alpha) * acc
        filtered[i] = acc
    return 0.05 * filtered


def _build_mono_float(sr: int) -> tuple[np.ndarray, list[dict]]:
    rng = _rng(MASTER_SEED)
    pieces: list[np.ndarray] = []
    segments: list[dict] = []
    cursor = 0
    for kind, seconds in SEGMENT_PLAN:
        n = int(round(seconds * sr))
        if kind == "dialogue":
            piece = _dialogue_segment(n, sr, rng)
        elif kind == "silence":
            piece = np.zeros(n, dtype=np.float64)
        elif kind == "room_noise":
            piece = _room_noise_segment(n, rng)
        else:  # pragma: no cover - 防御
            raise RepairError(E_FORMAT_UNSUPPORTED, f"unknown segment kind: {kind}", field="segment")
        pieces.append(piece)
        segments.append(
            {"kind": kind, "start_sample": cursor, "end_sample": cursor + n, "seconds": seconds}
        )
        cursor += n
    return np.concatenate(pieces).astype(np.float64), segments


def _to_stereo(mono: np.ndarray, sr: int) -> np.ndarray:
    """由同一主信号派生双声道。

    两声道承载同一主信号（等增益、无额外衰减/噪声），平均下混即还原主信号，
    从而“同一内容在 16bit 单声道与 24bit 双声道下”的修复结论可在声明容差内
    逐项对齐；通道处理逻辑（逐声道 vs 先下混）仍由引擎在双声道数组上验证。
    """
    stereo = np.stack([mono, mono], axis=1)
    return stereo


def _quantize_int(signal_float: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth not in config.SUPPORTED_BIT_DEPTHS:
        raise RepairError(
            E_PARAM_RANGE,
            f"bit depth must be one of {config.SUPPORTED_BIT_DEPTHS}, got {bit_depth}",
            field="bit_depth",
            context={"bit_depth": bit_depth},
        )
    if bit_depth == 16:
        max_val = float(2 ** 15 - 1)
        clipped = np.clip(signal_float, -1.0, 1.0)
        return np.round(clipped * max_val).astype(np.int16)
    # numpy 无 int24：先按 24bit 量化再回到 [-1,1] 浮点，由 soundfile 落 PCM_24，
    # 量化电平本身完全确定。
    max_val = float(2 ** 23 - 1)
    clipped = np.clip(signal_float, -1.0, 1.0)
    levels = np.round(clipped * max_val)
    return (levels / max_val).astype(np.float64)


def generate_material(
    out_dir: str,
    sr: int = config.SAMPLE_RATE,
) -> dict[str, str]:
    """生成两套素材 WAV 与段清单，返回文件路径字典。"""
    os.makedirs(out_dir, exist_ok=True)
    mono_f, segments = _build_mono_float(sr)
    stereo_f = _to_stereo(mono_f, sr)

    mono16_path = os.path.join(out_dir, "speech_mono_48k_16bit.wav")
    stereo24_path = os.path.join(out_dir, "speech_stereo_48k_24bit.wav")
    sf.write(mono16_path, _quantize_int(mono_f, 16), sr, subtype="PCM_16")
    sf.write(stereo24_path, _quantize_int(stereo_f, 24), sr, subtype="PCM_24")

    import json

    seg_path = os.path.join(out_dir, "segments.json")
    with open(seg_path, "w", encoding="utf-8") as fh:
        json.dump({"sample_rate": sr, "segments": segments}, fh, indent=2, sort_keys=True)

    return {
        "mono16": mono16_path,
        "stereo24": stereo24_path,
        "segments": seg_path,
    }


def file_sha256(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
