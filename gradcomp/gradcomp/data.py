"""确定性样本生成：初始参数与各工作进程本地梯度分片。

随机数全部来自 numpy.random.Generator(PCG64)，并通过 SeedSequence 从主种子
派生子流；相同 (seed, samples, dimension, workers) 组合重复运行得到逐位
一致的初始参数与梯度。每个训练步的梯度也由 (seed, rank, step) 决定，
因此快照恢复后续训可以逐位重放。
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .digest import array_digest


def _rng(*spawn_key: int) -> np.random.Generator:
    # PCG64 是 NumPy 默认 BitGenerator，跨平台数值稳定。
    return np.random.Generator(np.random.PCG64(
        np.random.SeedSequence(list(spawn_key))))


def initial_parameters(cfg: Config) -> np.ndarray:
    """所有工作进程共享的初始参数向量（逐位确定）。"""
    rng = _rng(cfg.seed, 0x5A5A, cfg.dimension)
    return rng.normal(loc=0.0, scale=0.5, size=cfg.dimension).astype(np.float64)


def local_gradient(cfg: Config, rank: int, step: int) -> np.ndarray:
    """rank 号工作进程在第 step 步（0 起）的本地梯度。

    用确定性伪数据模拟“样本数”影响：把 samples 个样本的逐样本梯度取平均，
    每个样本梯度由独立确定的 RNG 生成。
    """
    rng = _rng(cfg.seed, rank + 1, step, cfg.samples, cfg.dimension)
    total = np.zeros(cfg.dimension, dtype=np.float64)
    # 分块累加避免一次性大数组；顺序固定。
    batch = max(1, cfg.samples // 4)
    remaining = cfg.samples
    while remaining > 0:
        n = min(batch, remaining)
        total += rng.normal(size=(n, cfg.dimension)).sum(axis=0)
        remaining -= n
    mean = total / cfg.samples
    # 注入与 rank/step 相关的确定性结构，使不同工作进程梯度可区分。
    coord = np.arange(cfg.dimension, dtype=np.float64)
    signal = np.sin((rank + 1) * (coord + 1.0) / 7.0 + step * 0.37)
    return (mean + 0.25 * signal).astype(np.float64)


def all_local_gradients(cfg: Config, step: int) -> np.ndarray:
    """返回形状 (workers, dimension) 的全部本地梯度，rank 为行序。"""
    return np.stack([local_gradient(cfg, rank, step)
                     for rank in range(cfg.workers)], axis=0)


def gradient_shards(grads: np.ndarray, cfg: Config) -> list[list[np.ndarray]]:
    """把 (workers, dimension) 梯度按等长块切成分片。

    返回 shards[rank][chunk]；chunk 数量等于 workers。
    """
    size = cfg.chunk_size
    return [
        [grads[rank, c * size:(c + 1) * size].copy()
         for c in range(cfg.workers)]
        for rank in range(cfg.workers)
    ]


def data_summary(cfg: Config) -> dict:
    """初始参数与第 0 步各分片的数据摘要。"""
    params = initial_parameters(cfg)
    grads = all_local_gradients(cfg, step=0)
    shards = gradient_shards(grads, cfg)
    return {
        "parameter_digest": array_digest(params),
        "parameter_norm": float(np.linalg.norm(params)),
        "gradient_digests": [array_digest(grads[r]) for r in range(cfg.workers)],
        "shard_digests": [
            [array_digest(shards[r][c]) for c in range(cfg.workers)]
            for r in range(cfg.workers)
        ],
        "gradient_norms": [float(np.linalg.norm(grads[r]))
                           for r in range(cfg.workers)],
    }
