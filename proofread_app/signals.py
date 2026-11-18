"""可处理退出信号：完成当前样本阶段提交后以退出码 0 退出。"""
from __future__ import annotations

import signal
from dataclasses import dataclass


@dataclass
class SignalController:
    requested: bool = False
    signum: int = 0

    def _handler(self, signum, frame) -> None:  # noqa: ANN001
        self.requested = True
        self.signum = signum

    def install(self) -> None:
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def reset(self) -> None:
        self.requested = False
        self.signum = 0
