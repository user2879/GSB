"""真实子进程信号测试：SIGTERM 后退出码为 0 且状态完整。"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

from vrep.jsonio import read_json

from ._base import SampleTestBase


class SubprocessSignalTests(SampleTestBase):
    def test_sigterm_exit_code_zero(self) -> None:
        if os.name == "nt" or not hasattr(signal, "SIGTERM"):
            self.skipTest("SIGTERM 子进程测试仅支持 POSIX 环境")
        workspace = os.path.join(self.temp_root, "subproc-ws")
        helper = os.path.join(os.path.dirname(__file__), "_signal_subprocess.py")
        proc = subprocess.Popen(
            [sys.executable, helper, self.samples, workspace],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        ready_line = proc.stdout.readline()
        self.assertIn("ready", ready_line)
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 0, msg=stderr)
        self.assertIn("graceful-shutdown-ok", stdout)
        ledger = read_json(os.path.join(workspace, "store", "state", "ledger.json"))
        self.assertEqual(ledger["global_version"], 1)
        self.assertGreaterEqual(len(ledger["images"]), 1)
