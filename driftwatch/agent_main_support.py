from __future__ import annotations

import ctypes
import os


def parent_alive(parent_pid: int) -> bool:
    if parent_pid <= 0:
        return False
    if os.name == "nt":
        synchronize = 0x00100000
        wait_timeout = 0x00000102
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(synchronize, False, int(parent_pid))
        if not handle:
            return False
        try:
            result = kernel32.WaitForSingleObject(handle, 0)
            return result == wait_timeout
        finally:
            kernel32.CloseHandle(handle)
    try:
        if os.getppid() != parent_pid:
            return False
        os.kill(parent_pid, 0)
        return True
    except OSError:
        return False
