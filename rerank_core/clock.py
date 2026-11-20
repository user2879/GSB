"""本地时钟工具：默认使用单调钟实测延迟，另提供可注入时钟用于边界测试。"""

from __future__ import annotations

import time
from dataclasses import dataclass


class MonotonicClock:
    """基于 time.monotonic_ns 的真实单调时钟，不受系统时间回拨影响。"""

    def now_ns(self) -> int:
        return time.monotonic_ns()


class DeterministicClock:
    """测试用可控时钟：仅手动推进，不模拟任何真实延迟。"""

    def __init__(self, start_ns: int = 0):
        self._now = start_ns

    def now_ns(self) -> int:
        return self._now

    def advance_ns(self, delta_ns: int) -> None:
        self._now += delta_ns


@dataclass
class StageTimer:
    clock: MonotonicClock
    name: str
    started_ns: int = 0
    ended_ns: int = 0

    def __enter__(self) -> "StageTimer":
        self.started_ns = self.clock.now_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.ended_ns = self.clock.now_ns()

    @property
    def elapsed_ms(self) -> float:
        return (self.ended_ns - self.started_ns) / 1_000_000.0
