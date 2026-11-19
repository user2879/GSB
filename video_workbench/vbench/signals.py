"""可处理退出信号与强制终止注入的阶段边界闸门。

* 收到 SIGTERM/SIGINT 后，仅在阶段边界动作：让当前阶段产物完整落盘，
  随后提交暂存快照并以整数退出码 0 退出（不承诺生成额外退出报告）。
* ``hard_kill`` 注入用于模拟无法处理的强制终止（SIGKILL/断电）：
  在指定阶段产物落盘后立即 ``os._exit``，重启时仅靠已完整替换的
  产物与进度记录恢复。
"""
from __future__ import annotations

import os
import signal
from typing import Any, Mapping


class SignalGate:
    def __init__(
        self,
        injections: list[Mapping[str, Any]] | None = None,
        *,
        install_handlers: bool = True,
    ) -> None:
        self.injections = list(injections or [])
        self.requested_stop = False
        self.exited_cleanly = False
        self.hard_kill_stage: str | None = None
        self.hard_kill_id: str | None = None
        for injection in self.injections:
            if injection.get("type") == "hard_kill":
                self.hard_kill_stage = injection.get("stage")
                self.hard_kill_id = injection.get("id")
        self._previous: dict[int, Any] = {}
        if install_handlers:
            self._install()

    def _install(self) -> None:
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous[signum] = signal.signal(signum, self._handler)
            except (ValueError, OSError):
                # 非主线程或受限环境下无法注册处理器。
                continue

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                continue

    def _handler(self, signum: int, frame: Any) -> None:
        self.requested_stop = True

    def after_stage(self, stage_name: str, store: Any) -> None:
        if self.hard_kill_stage == stage_name:
            # 模拟 SIGKILL：不执行任何 Python 清理（缓冲、finally 均不运行）。
            os._exit(137)
        if self.requested_stop:
            store.commit()
            self.exited_cleanly = True
            raise SystemExit(0)
