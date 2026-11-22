"""确定性随机源：纯 Python 实现的 splitmix64 与比特流。

不使用 numpy.random，避免 NumPy 版本间随机算法差异，
保证同一整数种子在任何平台上逐位一致。
"""

from __future__ import annotations

import numpy as np

_MASK64 = (1 << 64) - 1


def splitmix64(state: int) -> tuple[int, int]:
    """splitmix64 步进，返回 (新状态, 64 位输出)。"""
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = z ^ (z >> 31)
    return state, z


def mix_seed(*parts: int) -> int:
    """把若干整数折叠为一个 64 位种子（顺序敏感）。"""
    state = 0xD1B54A32D192ED03
    for part in parts:
        state, _ = splitmix64(state ^ (part & _MASK64))
    return state


class BitStream:
    """从整数种子派生的确定性比特流。"""

    def __init__(self, seed: int) -> None:
        self._state = seed & _MASK64

    def next_uint64(self) -> int:
        self._state, value = splitmix64(self._state)
        return value

    def next_unit_float64(self) -> float:
        """返回 [0, 1) 双精度浮点，取高 53 位。"""
        return (self.next_uint64() >> 11) * (1.0 / (1 << 53))

    def float32_array(self, shape: tuple[int, ...]) -> np.ndarray:
        """生成 [-1, 1) 区间的 float32 数组，结果逐位确定。"""
        size = int(np.prod(shape))
        values = np.empty(size, dtype=np.float64)
        divisor = float(1 << 23)
        for index in range(size):
            unit = (self.next_uint64() >> 41) / divisor
            values[index] = 2.0 * unit - 1.0
        return np.ascontiguousarray(values.reshape(shape), dtype=np.float32)

    def bytes(self, length: int) -> bytes:
        chunks: list[bytes] = []
        remaining = length
        while remaining > 0:
            chunk = self.next_uint64().to_bytes(8, "big")[: min(8, remaining)]
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
