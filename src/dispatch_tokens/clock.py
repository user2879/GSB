"""单调时钟抽象。

令牌桶与到期判定只依赖单调毫秒，不依赖墙上时间。提供两种实现：

* :class:`DeterministicClock`：由事件流的 ``ts_ms`` 驱动，在线补算与重启
  重放使用同一实现，保证任意批次切分、任意真实等待时长下结论逐字节一致。
* :class:`MonotonicClock`：封装 :func:`time.monotonic_ns`，供节点子进程
  统计真实聚合耗时，以及真实套接字超时配合使用。
"""
from __future__ import annotations

import time
from typing import Iterable


class Clock:
    def now_ms(self) -> int:
        raise NotImplementedError


class DeterministicClock(Clock):
    """以显式毫秒时间戳推进的确定性时钟，时间只会前进。"""

    def __init__(self, start_ms: int = 0) -> None:
        self._now = int(start_ms)

    def now_ms(self) -> int:
        return self._now

    def advance_to(self, ts_ms: int) -> int:
        if ts_ms < self._now:
            raise ValueError(
                f"deterministic clock cannot move backwards: {ts_ms} < {self._now}")
        self._now = int(ts_ms)
        return self._now

    def advance_by(self, delta_ms: int) -> int:
        if delta_ms < 0:
            raise ValueError("delta_ms must be non-negative")
        self._now += int(delta_ms)
        return self._now


class MonotonicClock(Clock):
    """真实单调时钟，毫秒口径（纳秒整数除 1_000_000，向零截断）。"""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


def event_timeline(events: Iterable) -> list[int]:
    """提取事件时间轴，便于批量切分时复用。"""
    return [event.ts_ms for event in events]
