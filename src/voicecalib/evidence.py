"""初始证据强度构建：三维特征 -> 各情感强度 -> 稳定归一化 -> 定点整数。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CalibConfig
from .features import FEATURE_DIM, F0_IDX, ENERGY_IDX, RHYTHM_IDX, FrameFeatures
from .fixedpoint import FixedPoint
from .hashing import fingerprint, short_key

# 情感类别按字典序固定排列，保证所有向量分量顺序确定。
EMOTIONS = ("angry", "calm", "happy", "sad")

# 三维特征的固定参考范围，用于确定性归一化。
FEATURE_RANGES = ((70.0, 400.0), (0.0, 0.6), (0.0, 60.0))

# 情感原型矩阵（行=情感字典序，列=特征维度升序 f0/energy/rhythm）。
PROTOTYPES = np.array([
    [0.90, 0.85, 0.80],   # angry
    [0.30, 0.30, 0.25],   # calm
    [0.70, 0.60, 0.70],   # happy
    [0.20, 0.20, 0.15],   # sad
], dtype=np.float64)


@dataclass
class StrengthArtifact:
    round: int
    file: str
    raw_scores: list[float]
    softmax_float: list[float]
    strength_q: list[int]
    quant_residuals: list[float]
    feature_vector: list[float]
    fp_metadata: dict
    cache_key: str
    audio_hash: str
    sort_key: str

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "file": self.file,
            "emotions": list(EMOTIONS),
            "raw_scores": self.raw_scores,
            "softmax_float64": self.softmax_float,
            "strength_q": self.strength_q,
            "quant_residuals": self.quant_residuals,
            "feature_vector": self.feature_vector,
            "fixed_point": self.fp_metadata,
            "cache_key": self.cache_key,
            "audio_hash": self.audio_hash,
            "sort_key": self.sort_key,
        }


def normalize_features(vector: np.ndarray) -> np.ndarray:
    out = np.zeros(FEATURE_DIM, dtype=np.float64)
    # 特征维度升序的固定运算顺序。
    for dim in range(FEATURE_DIM):
        low, high = FEATURE_RANGES[dim]
        out[dim] = min(1.0, max(0.0, (float(vector[dim]) - low) / (high - low)))
    return out


def build_strength(features: FrameFeatures, audio_hash: str,
                   config: CalibConfig) -> StrengthArtifact:
    fp = FixedPoint(config.frac_bits, config.quant_gear)
    normalized = normalize_features(features.vector)

    # 加权原型距离：距离越小证据越强；固定顺序（情感字典序、维度升序）。
    raw_scores = [0.0] * len(EMOTIONS)
    for emo_idx in range(len(EMOTIONS)):
        acc = 0.0
        for dim in range(FEATURE_DIM):
            delta = normalized[dim] - PROTOTYPES[emo_idx, dim]
            acc += float(config.initial_weights[dim]) * delta * delta
        # 负距离作为证据得分，温度系数 0.15 固定，保证类别可分且确定。
        raw_scores[emo_idx] = -acc / 0.15

    # 减去最大值的稳定 softmax（归一化方式由配置声明，仅支持 softmax）。
    max_score = max(raw_scores)
    exps = [np.exp(score - max_score) for score in raw_scores]
    denom = sum(exps)
    softmax = [value / denom for value in exps]

    strength_q: list[int] = []
    residuals: list[float] = []
    for value in softmax:
        q, residual = fp.quantize(value)
        strength_q.append(q)
        residuals.append(residual)

    cache_key = short_key(
        "evidence",
        audio_hash,
        config.evidence_param_fingerprint,
        config.frac_bits,
        config.quant_gear,
        features.vector.tolist(),
    )
    return StrengthArtifact(
        round=features.round,
        file=features.file,
        raw_scores=raw_scores,
        softmax_float=softmax,
        strength_q=strength_q,
        quant_residuals=residuals,
        feature_vector=[float(v) for v in features.vector],
        fp_metadata=fp.metadata(),
        cache_key=cache_key,
        audio_hash=audio_hash,
        sort_key=features.sort_key,
    )


def evidence_fingerprint(artifact: StrengthArtifact) -> str:
    return fingerprint(artifact.to_dict())
