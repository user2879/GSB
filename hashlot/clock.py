"""单调时钟抽象。

生产环境使用 :class:`MonotonicClock`（基于 :func:`time.monotonic_ns`，
真实本地单调时钟），测试与可复现实验使用 :class:`FakeClock` 手动推进。
所有时间单位对外统一为整数毫秒。
"""

import time
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int: ...

    def sleep_ms(self, milliseconds: int) -> None: ...


class MonotonicClock:
    """真实本地单调时钟，返回整数毫秒。"""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000

    def sleep_ms(self, milliseconds: int) -> None:
        if milliseconds > 0:
            time.sleep(milliseconds / 1000.0)


class FakeClock:
    """虚拟单调时钟，仅能向前推进，保证测试可复现。"""

    def __init__(self, start_ms: int = 0):
        self._now = int(start_ms)

    def now_ms(self) -> int:
        return self._now

    def advance(self, milliseconds: int) -> None:
        if milliseconds < 0:
            raise ValueError("时钟只能向前推进")
        self._now += int(milliseconds)

    def sleep_ms(self, milliseconds: int) -> None:
        self.advance(milliseconds)
