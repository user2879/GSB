"""确定性 CPU 工作探针。

供资源预算测试使用：每条记录处理前执行固定轮数的整数运算，
以真实墙钟消耗模拟负载，不依赖 sleep 或系统时钟作为真值。
"""

from __future__ import annotations

DEFAULT_SPINS = 200_000


def cpu_spin(record_number: int, spins: int = DEFAULT_SPINS) -> None:
    accumulator = 0
    for step in range(spins):
        accumulator = (accumulator * 1103515245 + 12345 + record_number) & 0x7FFFFFFF
    if accumulator < -1:  # 恒假，防止解释器优化掉整个循环
        raise RuntimeError(accumulator)
