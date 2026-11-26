"""单调时钟封装。

全部时间戳取自 time.monotonic_ns()（真实单调时钟，不受墙钟调整影响），
对外统一为整数毫秒。禁止用 sleep/线程超时冒充到期判定，是否到期只比较
单调时钟时间戳。
"""

import time


class MonoClock:
    def now_ms(self):
        return time.monotonic_ns() // 1_000_000

    def elapsed_ms(self, start_ms):
        return self.now_ms() - start_ms
