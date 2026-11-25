"""本地确定性分类器：固定投影 -> 整数后验得分 -> 稳定 softmax 后验。"""

from dataclasses import dataclass

import numpy as np

from .config import ClassifierConfig, PipelineConfig, QuantConfig
from .errors import BusinessError, E_DIMENSION_MISMATCH, E_QUANT_LEVEL_INVALID
from .features import WindowFeature
from .fingerprint import PREFIX_WINDOW_POSTERIOR, cache_key, content_fingerprint


@dataclass(frozen=True)
class WindowPosterior:
    window_index: int
    scores: tuple[int, ...]          # 与命令编码升序对齐
    posteriors: tuple[float, ...]
    predicted_code: int
    predicted_word: str
    boundary: bool
    cache_key: str
    feature_key: str


def build_projection(cfg: PipelineConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.classifier.weight_seed)
    n_cmds = len(cfg.commands)
    matrix = rng.integers(
        -1, 2, size=(n_cmds, cfg.feature_dim), dtype=np.int64
    )
    return matrix


def _stable_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    shifted = (scores - scores.max()) / temperature
    exp_scores = np.exp(shifted.astype(np.float64))
    return exp_scores / exp_scores.sum()


def classify_window(
    feature: WindowFeature,
    cfg: PipelineConfig,
    projection: np.ndarray,
) -> WindowPosterior:
    if cfg.quant.quant_level not in cfg.quant.allowed_levels:
        raise BusinessError(
            E_QUANT_LEVEL_INVALID,
            f"量化档位 {cfg.quant.quant_level} 不在允许枚举内",
            stage="classify",
            window_index=feature.window_index,
        )
    vector = np.asarray(feature.fixed_point, dtype=np.int64)
    if vector.shape[0] != projection.shape[1]:
        raise BusinessError(
            E_DIMENSION_MISMATCH,
            f"后验输入维度 {vector.shape[0]} 与投影列数 {projection.shape[1]} 不符",
            stage="classify",
            window_index=feature.window_index,
        )
    scores = projection @ vector
    posteriors = _stable_softmax(scores, cfg.classifier.softmax_temperature)
    codes = sorted(cfg.vocab)
    order = sorted(
        range(len(codes)),
        key=lambda idx: (-float(posteriors[idx]), codes[idx]),
    )
    best, second = order[0], order[1]
    margin = float(posteriors[best] - posteriors[second])
    tied = float(posteriors[best]) == float(posteriors[second])
    boundary = bool(tied or margin <= cfg.classifier.boundary_margin)
    predicted_code = codes[best]
    param_fp = content_fingerprint(
        {
            "weight_seed": cfg.classifier.weight_seed,
            "boundary_margin": cfg.classifier.boundary_margin,
            "quant_level": cfg.quant.quant_level,
            "vocab": cfg.vocab_fingerprint(),
        }
    )
    key = cache_key(PREFIX_WINDOW_POSTERIOR, feature.cache_key, param_fp)
    return WindowPosterior(
        window_index=feature.window_index,
        scores=tuple(int(v) for v in scores),
        posteriors=tuple(float(v) for v in posteriors),
        predicted_code=predicted_code,
        predicted_word=cfg.vocab[predicted_code],
        boundary=boundary,
        cache_key=key,
        feature_key=feature.cache_key,
    )
