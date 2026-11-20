"""Lifecycle manager for the deterministic target-service child processes.

Services are tracked by PID (not Popen) so that processes orphaned by a
crashed orchestrator can be re-adopted on resume via their pidfiles and
still be suspended / killed / rebuilt for real.
"""

import os
import signal
import subprocess
import sys
import threading
import time

from .injectors import IS_WINDOWS, control_command


def _pid_alive(pid):
    if IS_WINDOWS:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _kill_pid(pid):
    if IS_WINDOWS:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x0001, False, pid)  # TERMINATE
        if handle:
            try:
                kernel32.TerminateProcess(handle, 1)
            finally:
                kernel32.CloseHandle(handle)
    else:
        os.kill(pid, signal.SIGKILL)


class ManagedProc:
    """A child (or adopted) service process identified by PID.

    Holds the Popen handle when we spawned the process ourselves so that
    poll()/wait() reap it (a reaped-less zombie would otherwise look alive
    to os.kill(pid, 0) forever)."""

    def __init__(self, pid, popen=None):
        self.pid = pid
        self._popen = popen

    def kill(self):
        if self._popen is not None:
            self._popen.kill()
        else:
            _kill_pid(self.pid)

    def wait(self, timeout=5):
        if self._popen is not None:
            return self._popen.wait(timeout=timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _pid_alive(self.pid):
                return 0
            time.sleep(0.02)
        raise subprocess.TimeoutExpired(self.pid, timeout)

    def poll(self):
        if self._popen is not None:
            return self._popen.poll()
        return None if _pid_alive(self.pid) else 0


class ServiceManager:
    def __init__(self, spec, log, run_dir):
        self.spec = spec
        self.log = log
        self.run_dir = str(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self._procs = {}
        self._lock = threading.Lock()

    def _pidfile(self, service_id):
        return os.path.join(self.run_dir, "%s.pid" % service_id)

    def _spawn(self, svc):
        args = [
            sys.executable, "-m", "chaos.service_sim",
            "--id", svc.id,
            "--port", str(svc.port),
            "--control-port", str(svc.control_port),
            "--latency-ms", str(svc.fixed_latency_ms),
            "--error-rate-ppm", str(svc.error_rate_ppm),
            "--seed", str(self.spec.seed),
            "--pidfile", self._pidfile(svc.id),
        ]
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        self._procs[svc.id] = ManagedProc(proc.pid, popen=proc)
        return self._procs[svc.id]

    def _wait_ready(self, svc, timeout_ms=5000):
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            try:
                if control_command(svc.control_port, "PING", timeout=0.5) == "PONG":
                    return True
            except OSError:
                time.sleep(0.05)
        return False

    def ensure_running(self):
        """Start every declared service; adopt live ones via pidfile."""
        for svc in self.spec.services:
            with self._lock:
                try:
                    alive = control_command(svc.control_port, "PING",
                                            timeout=0.3) == "PONG"
                except OSError:
                    alive = False
                if alive:
                    try:
                        with open(self._pidfile(svc.id)) as fh:
                            pid = int(fh.read().strip())
                        if _pid_alive(pid):
                            self._procs[svc.id] = ManagedProc(pid)
                            self.log.emit("service_adopted", service=svc.id,
                                          pid=pid)
                            continue
                    except (OSError, ValueError):
                        pass
                    self.log.emit("service_foreign", service=svc.id,
                                  note="alive but no usable pidfile")
                    continue
                self._spawn(svc)
                if not self._wait_ready(svc):
                    raise RuntimeError("service %s failed to start" % svc.id)
                self.log.emit("service_started", service=svc.id, port=svc.port)

    def proc_of(self, service_id):
        with self._lock:
            proc = self._procs.get(service_id)
        if proc is None:
            raise KeyError("no managed child process for %s" % service_id)
        return proc

    def respawn(self, service_id):
        svc = self.spec.service_by_id(service_id)
        with self._lock:
            proc = self._spawn(svc)
        if not self._wait_ready(svc):
            raise RuntimeError("service %s failed to rebuild" % service_id)
        self.log.emit("service_respawned", service=service_id, pid=proc.pid)
        return proc

    def stop_all(self):
        with self._lock:
            procs = list(self._procs.values())
            self._procs.clear()
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        for proc in procs:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
