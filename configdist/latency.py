"""交互延迟观测：单调时钟起止毫秒差，均值/P95/最大值定点表示。

每次界面查询与权威重算记录 time.monotonic_ns() 起止差（整数毫秒）。
统计量以 milli-ms 定点表示；量化误差上界：单次测量 <=0.5ms，
均值误差上界 <=0.5ms（最坏情形同向累积）。
"""

import threading
import time
from contextlib import contextmanager

QUANT_BOUND_MILLI_MS = 500  # 0.5ms，以 milli-ms 定点表示


class LatencyRecorder:
    def __init__(self):
        self._lock = threading.Lock()
        self._samples = {}

    @contextmanager
    def measure(self, kind):
        start = time.monotonic_ns()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            self.record(kind, elapsed_ms)

    def record(self, kind, elapsed_ms):
        with self._lock:
            self._samples.setdefault(kind, []).append(int(elapsed_ms))

    def stats(self, kind):
        with self._lock:
            samples = sorted(self._samples.get(kind, []))
        if not samples:
            return {"count": 0, "mean_milli_ms": 0, "p95_ms": 0,
                    "max_ms": 0, "quant_bound_milli_ms": QUANT_BOUND_MILLI_MS}
        n = len(samples)
        mean_milli = sum(samples) * 1000 // n
        p95_index = max(0, -(-95 * n // 100) - 1)  # ceil(0.95n)-1
        return {"count": n,
                "mean_milli_ms": mean_milli,
                "p95_ms": samples[p95_index],
                "max_ms": samples[-1],
                "quant_bound_milli_ms": QUANT_BOUND_MILLI_MS}

    def kinds(self):
        with self._lock:
            return sorted(self._samples)
