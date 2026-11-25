"""单调时钟封装。

系统中所有等待、租约计时都以单调时钟毫秒计量，避免墙钟回拨导致
租约被错误判定。测试可通过 :class:`FakeClock` 推进虚拟时间。
"""

import time


class MonotonicClock:
    """真实单调时钟，单位毫秒。"""

    def now_ms(self):
        return int(time.monotonic() * 1000.0)

    def sleep_ms(self, ms):
        if ms > 0:
            time.sleep(ms / 1000.0)


class FakeClock:
    """可控虚拟单调时钟，仅用于测试租约过期与等待计时。"""

    def __init__(self, start=0):
        self._now = int(start)

    def now_ms(self):
        return self._now

    def advance(self, ms):
        self._now += int(ms)
        return self._now

    def sleep_ms(self, ms):
        self.advance(ms)
