"""确定性随机源。

统一使用 numpy 的 PCG64 + 固定种子，保证两次运行逐字节一致；
禁止在素材生成路径使用系统时间或全局随机状态。
"""
from __future__ import annotations

import numpy as np

DEFAULT_SEED = 20261120


def np_rng(seed: int = DEFAULT_SEED, stream: int = 0) -> np.random.Generator:
    """返回独立流的确定性 Generator（同种子同流 => 同序列）。"""
    ss = np.random.SeedSequence([seed, stream])
    return np.random.Generator(np.random.PCG64(ss))


def channel_rng(channel: int, seed: int = DEFAULT_SEED) -> np.random.Generator:
    return np_rng(seed, stream=channel + 1)
