"""确定性 5.1 合成素材生成器。

固定 NumPy Generator 种子，程序化拼接对白段、音乐段与瞬态冲击段，输出
48kHz/24bit/6 声道 WAV 与分段真值标签。所有随机量来自同一确定性生成器，
不依赖时间或外部素材；同一配置两次生成逐字节一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .kweight import kweight_impulse, kweight_channels, CHANNEL_WEIGHTS, to_db_tp, true_peak_fir, upsample_max
from .meter import block_anchors, block_meansq, block_lufs, gated_lufs, percentile_lra
from .wavio import SAMPLE_RATE, CHANNELS_51, write_wav24

DEFAULT_SEED = 20261119
# 固定节目单：类型、时长（秒）、目标响度（LKFS，归一化后以实测为准）
DEFAULT_PROGRAM = [
    ("speech", 5.0, -24.0),
    ("music", 5.0, -20.0),
    ("transient", 2.0, -26.0),
    ("speech", 3.0, -23.0),
]
# 5.1 通道布局：L, R, C, LFE, Ls, Rs
SPEECH_LAYOUT = np.array([0.85, 0.85, 1.0, 0.0, 0.35, 0.35])
MUSIC_LAYOUT = np.array([1.0, 1.0, 0.55, 0.5, 0.9, 0.9])
TRANSIENT_LAYOUT = np.array([0.9, 0.9, 0.7, 1.0, 0.8, 0.8])


@dataclass
class Segment:
    kind: str
    start: int
    end: int
    target_lufs: float
    achieved_lufs: float
    lra_lu: float
    peak_linear: float
    true_peak_db: float
    sample_count: int
    source: str


def _speech(rng: np.random.Generator, n: int) -> np.ndarray:
    """对白：低频语调载波 + 音节包络（指数衰减脉冲串）+ 弱噪声。"""
    t = np.arange(n) / SAMPLE_RATE
    f0 = 120.0
    carrier = np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t)
    syllable_rate = 4.0
    phase = (syllable_rate * t) % 1.0
    envelope = np.exp(-6.0 * phase)
    pauses = (np.sin(2 * np.pi * (syllable_rate / 2.0) * t) > -0.2).astype(np.float64)
    noise = rng.standard_normal(n) * 0.05
    return (carrier * envelope * pauses + noise)


def _music(rng: np.random.Generator, n: int) -> np.ndarray:
    """音乐：确定频率集合的多音和弦 + 慢速 LFO + 低幅噪声。"""
    t = np.arange(n) / SAMPLE_RATE
    freqs = np.array([110.0, 164.81, 220.0, 329.63, 440.0])
    amps = np.array([1.0, 0.7, 0.6, 0.4, 0.3])
    out = np.zeros(n)
    for f, a in zip(freqs, amps):
        out += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    lfo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.5 * t)
    out = out / freqs.size * lfo
    out += rng.standard_normal(n) * 0.03
    return out


def _transient(rng: np.random.Generator, n: int) -> np.ndarray:
    """瞬态冲击：周期性指数衰减脉冲 + 稀疏强冲击。"""
    out = np.zeros(n)
    period = int(0.5 * SAMPLE_RATE)
    for k, center in enumerate(range(0, n, period)):
        local = np.arange(n - center)
        pulse = np.exp(-local / 40.0) * np.sin(2 * np.pi * 80.0 * local / SAMPLE_RATE)
        out[center:] += pulse
    # 一个明确的强冲击（但仍保留 -1 dBTP 以内余量）
    strong = int(1.25 * SAMPLE_RATE)
    if strong < n:
        local = np.arange(n - strong)
        out[strong:] += 2.5 * np.exp(-local / 25.0) * np.sin(2 * np.pi * 120.0 * local / SAMPLE_RATE)
    out += rng.standard_normal(n) * 0.02
    return out


_GENERATORS = {"speech": _speech, "music": _music, "transient": _transient}
_LAYOUTS = {"speech": SPEECH_LAYOUT, "music": MUSIC_LAYOUT, "transient": TRANSIENT_LAYOUT}


def _normalize_segment(mono: np.ndarray, layout: np.ndarray, target_lufs: float) -> Tuple[np.ndarray, float]:
    """按 BS.1770 K 加权门控响度把段落归一到目标响度（带峰值保护）。"""
    seg = mono[:, None] * layout[None, :]
    seg = seg.T.copy()  # (channels, samples)
    impulse = kweight_impulse()
    weighted = kweight_channels(seg, impulse)
    anchors = block_anchors(seg.shape[1])
    if anchors:
        ms = [block_meansq(weighted, a) for a in anchors]
        _, current, _, _ = gated_lufs(ms)
        current = current if current is not None else block_lufs(float(np.mean(ms)))
    else:
        rms = np.mean(np.sum(seg * seg * CHANNEL_WEIGHTS.reshape(-1, 1), axis=0))
        current = block_lufs(float(rms))
    gain = 10.0 ** ((target_lufs - current) / 20.0)
    peak = float(np.max(np.abs(seg))) * gain
    if peak > 0.89:  # 峰值保护，保证 24bit 量化后真峰值不超过 0 dBTP
        gain *= 0.89 / peak
    return seg * gain, gain


def _segment_truth(seg: np.ndarray, kind: str, target: float, achieved: float, gain: float) -> Dict:
    impulse = kweight_impulse()
    weighted = kweight_channels(seg, impulse)
    anchors = block_anchors(seg.shape[1])
    ms = [block_meansq(weighted, a) for a in anchors]
    lufs_vals = [block_lufs(m) for m in ms]
    _, rel_lufs, _, _ = gated_lufs(ms)
    lra = percentile_lra(lufs_vals)
    fir = true_peak_fir()
    tp = -np.inf
    for c in range(seg.shape[0]):
        p, _ = upsample_max(seg[c], fir)
        tp = max(tp, to_db_tp(p))
    return {
        "achieved_gated_lufs": rel_lufs,
        "lra_lu": lra,
        "peak_linear": float(np.max(np.abs(seg))),
        "true_peak_db": float(tp),
        "normalization_gain": gain,
    }


def generate_program(
    seed: int = DEFAULT_SEED,
    program: List[Tuple[str, float, float]] = None,
) -> Tuple[np.ndarray, List[Segment], Dict]:
    program = list(program or DEFAULT_PROGRAM)
    rng = np.random.default_rng(seed)
    pieces: List[np.ndarray] = []
    segments: List[Segment] = []
    cursor = 0
    for kind, seconds, target in program:
        n = int(round(seconds * SAMPLE_RATE))
        mono = _GENERATORS[kind](rng, n)
        seg, gain = _normalize_segment(mono, _LAYOUTS[kind], target)
        truth = _segment_truth(seg, kind, target, target, gain)
        segments.append(
            Segment(
                kind=kind,
                start=cursor,
                end=cursor + n,
                target_lufs=float(target),
                achieved_lufs=truth["achieved_gated_lufs"],
                lra_lu=truth["lra_lu"],
                peak_linear=truth["peak_linear"],
                true_peak_db=truth["true_peak_db"],
                sample_count=n,
                source=f"deterministic:{kind}:seed={seed}:gain={gain:.12f}",
            )
        )
        pieces.append(seg)
        cursor += n

    audio = np.concatenate(pieces, axis=1).astype(np.float64)
    assert audio.shape == (CHANNELS_51, cursor)
    meta = {
        "schema": "loudbench.synth-labels/v1",
        "seed": seed,
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS_51,
        "sample_width_bits": 24,
        "total_samples": cursor,
        "duration_seconds": cursor / SAMPLE_RATE,
        "ground_truth_source": (
            "所有段落由 numpy.default_rng(seed) 程序化生成；目标响度按 BS.1770 "
            "K加权两级门控实测后归一，achieved_* 为量化前 float64 真值，"
            "真峰值为 4x 过采样估计。无外部素材、无时间随机量。"
        ),
        "segments": [vars(sg) for sg in segments],
    }
    return audio, segments, meta


def write_program(workdir: str, seed: int = DEFAULT_SEED) -> Dict[str, str]:
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    wav_path = work / "sample_5ch1_48k24.wav"
    label_path = work / "ground_truth_labels.json"
    audio, _, meta = generate_program(seed=seed)
    write_wav24(str(wav_path), audio, SAMPLE_RATE)
    label_path.write_text(
        json.dumps(meta, sort_keys=True, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"wav": str(wav_path), "labels": str(label_path)}
