"""Real injection mechanisms and verified reverts.

Every injector exposes apply() / revert(). revert() actively *verifies* the
effect is gone (probe the proxy port, query control STATUS, inspect process
state) instead of assuming so after a sleep. Verification failure raises
ChaosError(E_REVERT) with evidence preserved.
"""

import os
import socket
import subprocess
import time

from .errors import ChaosError, E_REVERT

IS_WINDOWS = os.name == "nt"


def _connect_probe(port, timeout=0.5):
    """Return True if a TCP connect to 127.0.0.1:port succeeds."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for(predicate, timeout_ms, poll_ms=25):
    """Poll predicate until true or deadline. Returns final value."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_ms / 1000.0)
    return predicate()


def control_command(port, line, timeout=2.0):
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as conn:
        conn.sendall(line.encode() + b"\n")
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = conn.recv(1024)
            if not chunk:
                break
            buf += chunk
    return buf.decode().strip()


class LatencyProxy:
    """Loopback TCP proxy that adds a fixed one-way delay per request."""

    def __init__(self, target_port, delay_ms):
        self.target_port = target_port
        self.delay_ms = delay_ms
        self.proxy_port = None
        self._server = None
        self._thread = None

    def apply(self):
        import socketserver
        delay_s = self.delay_ms / 1000.0
        upstream = self.target_port

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                try:
                    up = socket.create_connection(("127.0.0.1", upstream), timeout=2)
                except OSError:
                    return
                with up:
                    data = self.request.recv(4096)
                    if not data:
                        return
                    time.sleep(delay_s)
                    up.sendall(data)
                    resp = b""
                    while not resp.endswith(b"\n"):
                        chunk = up.recv(4096)
                        if not chunk:
                            break
                        resp += chunk
                    self.request.sendall(resp)

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server(("127.0.0.1", 0), Handler)
        self.proxy_port = self._server.server_address[1]
        import threading
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        if not _wait_for(lambda: _connect_probe(self.proxy_port), 2000):
            raise ChaosError(E_REVERT, "latency proxy failed to listen",
                             evidence={"proxy_port": self.proxy_port})
        return self.proxy_port

    def revert(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        # Verify the injection is really gone: proxy port must refuse connects.
        if _wait_for(lambda: not _connect_probe(self.proxy_port), 2000):
            return
        raise ChaosError(E_REVERT, "latency proxy port still open after shutdown",
                         evidence={"proxy_port": self.proxy_port})


class ErrorRateOverride:
    def __init__(self, control_port, ppm):
        self.control_port = control_port
        self.ppm = ppm

    def apply(self):
        resp = control_command(self.control_port, f"SET_ERR {self.ppm}")
        if resp != "OK":
            raise ChaosError(E_REVERT, f"SET_ERR rejected: {resp}",
                             evidence={"response": resp})
        status = control_command(self.control_port, "STATUS")
        if f"override={self.ppm}" not in status:
            raise ChaosError(E_REVERT, "override not visible after SET_ERR",
                             evidence={"status": status})

    def revert(self):
        resp = control_command(self.control_port, "CLEAR_ERR")
        if resp != "OK":
            raise ChaosError(E_REVERT, f"CLEAR_ERR rejected: {resp}",
                             evidence={"response": resp})
        status = control_command(self.control_port, "STATUS")
        if "override=-1" not in status:
            raise ChaosError(E_REVERT, "error override still active after CLEAR_ERR",
                             evidence={"status": status})


def _proc_suspended(pid):
    if IS_WINDOWS:
        return None  # best-effort: no portable query; rely on call success
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] == "T"
    except OSError:
        return None


def _suspend_pid(pid):
    if IS_WINDOWS:
        import ctypes
        ntdll = ctypes.windll.ntdll
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x0800, False, pid)  # PROCESS_SUSPEND_RESUME
        if not handle:
            raise OSError("OpenProcess failed")
        try:
            if ntdll.NtSuspendProcess(handle) != 0:
                raise OSError("NtSuspendProcess failed")
        finally:
            kernel32.CloseHandle(handle)
    else:
        import signal
        os.kill(pid, signal.SIGSTOP)


def _resume_pid(pid):
    if IS_WINDOWS:
        import ctypes
        ntdll = ctypes.windll.ntdll
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x0800, False, pid)
        if not handle:
            raise OSError("OpenProcess failed")
        try:
            if ntdll.NtResumeProcess(handle) != 0:
                raise OSError("NtResumeProcess failed")
        finally:
            kernel32.CloseHandle(handle)
    else:
        import signal
        os.kill(pid, signal.SIGCONT)


class SuspendProcess:
    def __init__(self, pid, control_port):
        self.pid = pid
        self.control_port = control_port

    def apply(self):
        _suspend_pid(self.pid)
        if not _wait_for(lambda: _proc_suspended(self.pid) is not False, 2000):
            raise ChaosError(E_REVERT, "process not in stopped state after suspend",
                             evidence={"pid": self.pid})

    def revert(self):
        _resume_pid(self.pid)
        if not _wait_for(lambda: _proc_suspended(self.pid) is not True, 2000):
            raise ChaosError(E_REVERT, "process still stopped after resume",
                             evidence={"pid": self.pid})
        # Functional verification: control plane must answer again.
        if not _wait_for(lambda: _connect_probe(self.control_port), 2000):
            raise ChaosError(E_REVERT, "service control port unreachable after resume",
                             evidence={"pid": self.pid})


class KillProcess:
    """Terminate the child process; revert = rebuild per declaration or
    confirm it is not rebuildable (evidence either way)."""

    def __init__(self, service_manager, service_id, restart):
        self.mgr = service_manager
        self.service_id = service_id
        self.restart = restart

    def apply(self):
        proc = self.mgr.proc_of(self.service_id)
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            raise ChaosError(E_REVERT, "process did not exit after kill",
                             evidence={"pid": proc.pid})
        svc = self.mgr.spec.service_by_id(self.service_id)
        if _connect_probe(svc.port):
            raise ChaosError(E_REVERT, "data port still open after kill",
                             evidence={"pid": proc.pid, "port": svc.port})

    def revert(self):
        svc = self.mgr.spec.service_by_id(self.service_id)
        if self.restart and svc.rebuildable:
            self.mgr.respawn(self.service_id)
            if not _wait_for(lambda: _connect_probe(svc.control_port), 5000):
                raise ChaosError(E_REVERT, "rebuilt service did not come back",
                                 evidence={"service": self.service_id})
            return
        # Declared not rebuildable: confirm it stays down and keep evidence.
        if _connect_probe(svc.port) or _connect_probe(svc.control_port):
            raise ChaosError(E_REVERT, "non-rebuildable service unexpectedly alive",
                             evidence={"service": self.service_id})
