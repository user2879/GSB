"""真实资源预算：墙钟超时与单文件记录数上限。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from .errors import E_BUDGET, E_TIMEOUT


@dataclass(frozen=True)
class BudgetConfig:
    concurrency: int = 1
    per_file_timeout_seconds: float = 30.0
    max_records_per_file: int = 100_000


class BudgetGuard:
    """在记录处理循环中做真实墙钟判定与预算判定。"""

    def __init__(self, config: BudgetConfig, work_probe: Optional[Callable[[int], None]] = None):
        self.config = config
        self.started = time.monotonic()
        self.work_probe = work_probe
        self.stop_code: Optional[str] = None

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def check(self, record_number: int) -> Optional[str]:
        """处理下一条记录前调用；返回阻断码或 None。"""
        if self.stop_code is not None:
            return self.stop_code
        if record_number >= self.config.max_records_per_file:
            self.stop_code = E_BUDGET
            return E_BUDGET
        if self.elapsed() > self.config.per_file_timeout_seconds:
            self.stop_code = E_TIMEOUT
            return E_TIMEOUT
        if self.work_probe is not None:
            self.work_probe(record_number)
            if self.elapsed() > self.config.per_file_timeout_seconds:
                self.stop_code = E_TIMEOUT
                return E_TIMEOUT
        return None
