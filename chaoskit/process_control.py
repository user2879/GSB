"""Cross-platform real process suspend/resume and forced termination.

Windows suspends the whole process through ``NtSuspendProcess`` (kernel32's
``DebugActiveProcess`` would trigger debugger semantics), while POSIX uses
``SIGSTOP``/``SIGCONT``. Termination uses ``TerminateProcess`` on Windows and
``SIGKILL`` elsewhere. These change only process state, never source code.
"""
from __future__ import annotations

import signal
import subprocess
import sys

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:  # pragma: no cover - exercised on Windows hosts
    import ctypes
    from ctypes import wintypes

    _ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _NT_SUCCESS = 0

    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _ntdll.NtSuspendProcess.restype = wintypes.LONG
    _ntdll.NtSuspendProcess.argtypes = [wintypes.HANDLE]
    _ntdll.NtResumeProcess.restype = wintypes.LONG
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]

    PROCESS_SUSPEND_RESUME = 0x0800
    PROCESS_TERMINATE = 0x0001


def suspend_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if _IS_WINDOWS:  # pragma: no cover
        handle = _kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, proc.pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            status = _ntdll.NtSuspendProcess(handle)
            if status != _NT_SUCCESS:
                raise OSError(f"NtSuspendProcess failed with status {status:#x}")
        finally:
            _kernel32.CloseHandle(handle)
    else:
        proc.send_signal(signal.SIGSTOP)


def resume_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if _IS_WINDOWS:  # pragma: no cover
        handle = _kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, proc.pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            status = _ntdll.NtResumeProcess(handle)
            if status != _NT_SUCCESS:
                raise OSError(f"NtResumeProcess failed with status {status:#x}")
        finally:
            _kernel32.CloseHandle(handle)
    else:
        proc.send_signal(signal.SIGCONT)


def force_terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if _IS_WINDOWS:  # pragma: no cover
        proc.kill()  # subprocess uses TerminateProcess on Windows
    else:
        proc.kill()  # SIGKILL on POSIX
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def is_alive(proc: subprocess.Popen) -> bool:
    return proc.poll() is None


def is_suspended(proc: subprocess.Popen) -> bool:
    """Best-effort suspended-state probe.

    On Linux a stopped process reports state ``T`` in ``/proc``. On Windows the
    suspend count is not exposed cheaply, so the orchestrator tracks the
    intended state itself and this helper only distinguishes alive vs dead;
    the state column in the ledger remains the authoritative recorded truth.
    """
    if not is_alive(proc):
        return False
    if _IS_WINDOWS:  # pragma: no cover
        return False
    try:
        with open(f"/proc/{proc.pid}/stat", "r", encoding="ascii") as handle:
            data = handle.read()
        # comm is parenthesised and may contain spaces/parens; state follows.
        state = data[data.rfind(")") + 2]
        return state == "T"
    except OSError:
        return False
