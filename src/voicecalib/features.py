"""逐帧声学证据提取：基频统计、能量包络、语速节奏三维特征。

帧长 25ms / 帧移 10ms（可在配置中覆盖）。所有计算只依赖输入采样与帧序号，
不使用任何全局随机状态；任务可在线程池中并发执行。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE

FEATURE_DIM = 3
F0_IDX, ENERGY_IDX, RHYTHM_IDX = 0, 1, 2
FEATURE_NAMES = ("f0_mean_hz", "energy_rms", "rhythm_rate")


@dataclass
class FrameFeatures:
    round: int
    file: str
    n_frames: int
    vector: np.ndarray          # shape (3,) float64
    frame_f0: np.ndarray        # shape (n_frames,)
    frame_energy: np.ndarray
    voiced_ratio: float
    sort_key: str

    def to_summary_dict(self) -> dict:
        return {
            "round": self.round,
            "file": self.file,
            "n_frames": self.n_frames,
            "voiced_ratio": self.voiced_ratio,
            "features": {
                FEATURE_NAMES[F0_IDX]: float(self.vector[F0_IDX]),
                FEATURE_NAMES[ENERGY_IDX]: float(self.vector[ENERGY_IDX]),
                FEATURE_NAMES[RHYTHM_IDX]: float(self.vector[RHYTHM_IDX]),
            },
        }


def frame_bounds(n_samples: int, frame_length_ms: float,
                 frame_shift_ms: float) -> list[tuple[int, int]]:
    fl = int(round(SAMPLE_RATE * frame_length_ms / 1000.0))
    fs = int(round(SAMPLE_RATE * frame_shift_ms / 1000.0))
    bounds = []
    start = 0
    while start + fl <= n_samples:
        bounds.append((start, start + fl))
        start += fs
    return bounds


F0_MIN_HZ = 70.0
F0_MAX_HZ = 400.0


def _frame_f0(frame: np.ndarray) -> float:
    """归一化自相关基频估计（确定性，无外部依赖）。

    在 [F0_MIN_HZ, F0_MAX_HZ] 对应滞后区间取自相关主峰，
    对谐波与噪声比过零率稳健，适合区分不同说话人的基频均值。
    """
    n = frame.shape[0]
    if n < SAMPLE_RATE / F0_MIN_HZ:
        return 0.0
    centered = frame - float(np.mean(frame))
    energy = float(np.dot(centered, centered))
    if energy <= 1e-12:
        return 0.0
    lag_min = max(1, int(SAMPLE_RATE / F0_MAX_HZ))
    lag_max = min(n - 1, int(SAMPLE_RATE / F0_MIN_HZ))
    # 滞后升序扫描，首个超过 0.5 阈值的峰即基频周期（固定运算顺序）。
    best_lag = 0
    best_corr = 0.5
    for lag in range(lag_min, lag_max + 1):
        corr = float(np.dot(centered[:n - lag], centered[lag:])) / energy
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    if best_lag == 0:
        return 0.0
    return SAMPLE_RATE / best_lag


def extract_features(round_index: int, file_name: str, samples: np.ndarray,
                     frame_length_ms: float, frame_shift_ms: float,
                     sort_key: str) -> FrameFeatures:
    bounds = frame_bounds(samples.shape[0], frame_length_ms, frame_shift_ms)
    n_frames = len(bounds)
    frame_f0 = np.zeros(n_frames, dtype=np.float64)
    frame_energy = np.zeros(n_frames, dtype=np.float64)
    # 固定帧序号升序计算，杜绝归约顺序歧义。
    for idx, (start, end) in enumerate(bounds):
        frame = samples[start:end]
        frame_energy[idx] = math.sqrt(float(np.mean(frame * frame)))
        frame_f0[idx] = _frame_f0(frame)

    voiced = frame_energy > (float(np.max(frame_energy)) * 0.2
                             if n_frames else 0.0)
    voiced_values = frame_f0[voiced]
    f0_mean = float(np.mean(voiced_values)) if voiced_values.size else 0.0
    energy_rms = float(math.sqrt(np.mean(frame_energy * frame_energy))) \
        if n_frames else 0.0
    if n_frames >= 2:
        envelope_diff = np.abs(np.diff(frame_energy))
        # 节奏：每秒能量包络显著起伏次数（固定阈值 0.02）。
        rhythm_rate = float(np.count_nonzero(envelope_diff > 0.02)) / (
            n_frames * frame_shift_ms / 1000.0)
    else:
        rhythm_rate = 0.0

    vector = np.array([f0_mean, energy_rms, rhythm_rate], dtype=np.float64)
    return FrameFeatures(
        round=round_index,
        file=file_name,
        n_frames=n_frames,
        vector=vector,
        frame_f0=frame_f0,
        frame_energy=frame_energy,
        voiced_ratio=float(np.count_nonzero(voiced) / n_frames) if n_frames else 0.0,
        sort_key=sort_key,
    )


def make_sort_key(file_name: str, stage: str) -> str:
    # 稳定排序键由文件名与阶段名派生（帧任务以帧序号后缀区分）。
    return f"{stage}|{file_name}"
