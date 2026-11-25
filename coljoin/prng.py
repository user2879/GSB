"""固定整数种子驱动的确定性伪随机源。

使用 SplitMix64（无外部状态依赖、跨平台位级一致），
样本生成只依赖整数种子与调用次序，不依赖网络或系统熵。
"""

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int):
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed 必须为非负整数")
        self._state = seed & MASK64

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def next_int(self, low: int, high: int) -> int:
        """闭区间 [low, high] 上的均匀整数。"""
        if high < low:
            raise ValueError("上界不得小于下界")
        span = high - low + 1
        return low + (self.next_u64() % span)

    def next_float(self, low: float = 0.0, high: float = 1.0) -> float:
        unit = self.next_u64() / float(1 << 64)
        return low + unit * (high - low)

    def choice(self, items):
        if not items:
            raise ValueError("空序列无法 choice")
        return items[self.next_u64() % len(items)]
