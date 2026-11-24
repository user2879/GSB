"""单调时钟抽象。

* :class:`MonotonicClock` 直接读取 :func:`time.monotonic_ns`，仲裁服务真实运行时使用；
* :class:`LogicalClock` 由调用方推进，确定性重放 / 独立校验使用，保证逐字节可复现。

两类时钟都以整数毫秒返回，租约到期判定只允许基于该单调毫秒与实际回执往返，
不允许通过线程 ``sleep`` 模拟超时（``sleep`` 仅用于按事件到达节拍推进调度）。
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int: ...


class MonotonicClock:
    """真实本地单调时钟，分辨率为纳秒，输出整数毫秒（向下取整）。"""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


class LogicalClock:
    """可显式推进的逻辑单调时钟，供确定性引擎与校验器重放。"""

    def __init__(self, start_ms: int = 0) -> None:
        self._now = int(start_ms)

    def now_ms(self) -> int:
        return self._now

    def advance_to(self, target_ms: int) -> None:
        target_ms = int(target_ms)
        if target_ms < self._now:
            raise ValueError("逻辑时钟不能严格回拨到已观测时刻之前")
        self._now = target_ms


    def advance_by(self, delta_ms: int) -> None:
        if delta_ms < 0:
            raise ValueError("逻辑时钟增量不能为负")
        self._now += int(delta_ms)
