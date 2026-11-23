"""Deterministic, dependency free pseudo random generator.

SplitMix64 gives the same integer stream on every platform and Python
version, which is what byte-for-byte reproducible alert generation needs.
"""

MASK64 = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int):
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def randrange(self, n: int) -> int:
        """Uniform integer in [0, n) without modulo bias."""
        if n <= 0:
            raise ValueError("randrange bound must be positive")
        limit = MASK64 - (MASK64 % n)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % n

    def random_unit(self) -> float:
        return self.next_u64() / (1 << 64)

    def shuffled(self, values: list) -> list:
        """Fisher-Yates shuffle returning a new list."""
        result = list(values)
        for i in range(len(result) - 1, 0, -1):
            j = self.randrange(i + 1)
            result[i], result[j] = result[j], result[i]
        return result

    def permutation_indices(self, n: int) -> list[int]:
        return self.shuffled(list(range(n)))
