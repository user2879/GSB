"""确定性随机数源。

使用纯 Python 实现的 SplitMix64（与 numpy 版本无关），保证场景生成在
任何机器、任何 numpy 版本上逐字节一致。序列只依赖固定种子与调用顺序。
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1


class DeterministicRng:
    def __init__(self, seed: int):
        self.seed = int(seed) & MASK64
        self._state = self.seed

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def next_unit(self) -> float:
        """[0, 1) 上的确定性双精度浮点数。"""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self.next_unit()

    def uniform_vec3(self, low: float, high: float) -> tuple[float, float, float]:
        return (
            self.uniform(low, high),
            self.uniform(low, high),
            self.uniform(low, high),
        )
