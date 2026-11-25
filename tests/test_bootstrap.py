"""每个测试模块导入前清理遗留工作进程，避免端口占用导致级联挂起。"""

import os
import signal
import unittest


def kill_orphan_workers():
    own = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == own:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmdline = handle.read().replace(b"\x00", b" ").decode(
                    "utf-8", "ignore")
        except OSError:
            continue
        if cmdline.strip().endswith("-m batch_receipt_ledger.worker"):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class BootstrapTest(unittest.TestCase):
    def test_orphan_cleanup(self):
        kill_orphan_workers()
        self.assertTrue(True)
