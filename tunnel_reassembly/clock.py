"""时钟抽象。

真实运行使用 MonotonicClock（time.monotonic_ns，毫秒整数）。
确定性重放使用 ReplayClock：按逐片记录的到达时刻顺序回放，
使同一脚本多次回放得到逐位一致的偏差估计与窗口序列。
"""

from __future__ import annotations

import time


class Clock:
    """时钟接口，返回整数毫秒。"""

    def now_ms(self) -> int:
        raise NotImplementedError


class MonotonicClock(Clock):
    """本地单调时钟毫秒。不参与跨进程可比性，仅用于单次真实接收。"""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


class ReplayClock(Clock):
    """确定性时钟：按给定到达时间序列依次返回。

    构造参数为严格非降的整数毫秒序列；now_ms 每次调用消费下一个值。
    若调用次数超过序列长度，固定返回最后一个值加一，保证单调性。
    """

    def __init__(self, arrive_ms_sequence: list[int]) -> None:
        self._seq = list(arrive_ms_sequence)
        self._idx = 0
        self._tail_extra = 0

    def now_ms(self) -> int:
        if self._idx < len(self._seq):
            value = self._seq[self._idx]
            self._idx += 1
            return value
        self._tail_extra += 1
        return (self._seq[-1] if self._seq else 0) + self._tail_extra
