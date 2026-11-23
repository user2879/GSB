"""确定性随机源：所有随机数来自固定种子的 NumPy Generator。

禁止使用全局 np.random，保证同一 (种子, 用途键) 下逐元素一致。
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEED = 20261123


def root_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def child_rng(rng: np.random.Generator, purpose: str) -> np.random.Generator:
    """从父 Generator 派生稳定子流，用途键决定具体序列。"""

    seed_seq = np.random.SeedSequence(
        [int(rng.integers(0, 2**31 - 1)), _purpose_hash(purpose)]
    )
    return np.random.default_rng(seed_seq)


def _purpose_hash(purpose: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.blake2b(purpose.encode(), digest_size=4).digest(), "big")
