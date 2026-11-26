"""固定整数种子的确定性 LCG，全项目统一使用，保证可重复演示。"""

from __future__ import annotations


class DeterministicRng:
    MOD = 2 ** 31
    MULT = 1103515245
    INCR = 12345

    def __init__(self, seed: int) -> None:
        self.state = int(seed) % self.MOD

    def next_u32(self) -> int:
        self.state = (self.MULT * self.state + self.INCR) % self.MOD
        return self.state

    def below(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        return self.next_u32() % bound

    def choose(self, options):
        return options[self.below(len(options))]

    def child(self, salt: int) -> "DeterministicRng":
        return DeterministicRng((self.state ^ (0x9E3779B9 * (salt + 1))) % self.MOD)
