from __future__ import annotations


MASK64 = (1 << 64) - 1
GOLDEN_GAMMA = 0x9E3779B97F4A7C15


class SplitMix64:
    """Small deterministic PRNG whose sequence is independent of Python's random module."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + GOLDEN_GAMMA) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def randbelow(self, n: int) -> int:
        if n <= 0:
            raise ValueError("n must be positive")
        return self.next_u64() % n

    def randint(self, low: int, high: int) -> int:
        if high < low:
            raise ValueError("high must be greater than or equal to low")
        return low + self.randbelow(high - low + 1)

    def sample_indices(self, population: int, k: int) -> list[int]:
        if k < 0 or k > population:
            raise ValueError("k outside population")
        values = list(range(population))
        for i in range(k):
            j = i + self.randbelow(population - i)
            values[i], values[j] = values[j], values[i]
        return sorted(values[:k])
