from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import DiagnosticError, ErrorCode
from .protocol import WorkerClient
from .spec import DatabaseSpec, Spec


def _timeout_error(db_id: str, message: str) -> DiagnosticError:
    return DiagnosticError(ErrorCode.TIMEOUT.value, message, db_id=db_id)


def _port_error(db_id: str, message: str) -> DiagnosticError:
    return DiagnosticError(ErrorCode.PORT.value, message, db_id=db_id)


class DatabaseRuntime:
    """Owns one real child process and its loopback TCP endpoint."""

    def __init__(
        self,
        spec_path: str | Path,
        database: DatabaseSpec,
        *,
        startup_timeout_seconds: float = 3.0,
        socket_timeout_seconds: float = 2.0,
    ) -> None:
        self.spec_path = str(Path(spec_path).resolve())
        self.database = database
        self.startup_timeout_seconds = startup_timeout_seconds
        self.socket_timeout_seconds = socket_timeout_seconds
        self.process: subprocess.Popen[str] | None = None
        self._stderr_text = ""

    @property
    def client(self) -> WorkerClient:
        return WorkerClient(self.database, timeout_seconds=self.socket_timeout_seconds)

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _read_child_output(self) -> str:
        if self.process is None:
            return ""
        try:
            stdout, stderr = self.process.communicate(timeout=0.2)
        except subprocess.TimeoutExpired:
            return ""
        self._stderr_text = stderr or self._stderr_text
        return stdout or ""

    def _raise_child_failure(self, last_error: Exception | None = None) -> None:
        if self.process is None:
            return
        output = self._read_child_output()
        fatal: dict[str, Any] | None = None
        for line in output.splitlines():
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and candidate.get("type") == "fatal":
                fatal = candidate
                break
        if fatal is not None and fatal.get("code") == ErrorCode.PORT.value:
            raise _port_error(self.database.id, str(fatal.get("message", "port bind failed")))
        if fatal is not None and fatal.get("code") == ErrorCode.SPEC.value:
            raise DiagnosticError(
                ErrorCode.SPEC.value,
                str(fatal.get("message", "worker rejected the specification")),
                db_id=self.database.id,
                field="spec",
            )
        noisy = (output + "\n" + self._stderr_text).strip()
        raise _port_error(
            self.database.id,
            f"worker {self.database.id!r} exited during startup with code "
            f"{self.process.returncode}: {noisy or 'no child output'}",
        )

    def start(self) -> None:
        if self.running:
            return
        command = [
            sys.executable,
            "-u",
            "-m",
            "shard_diag.worker",
            "--spec",
            self.spec_path,
            "--db-id",
            self.database.id,
        ]
        environment = os.environ.copy()
        package_root = str(Path(__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = package_root + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.startup_timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.process is None:
                raise _port_error(
                    self.database.id,
                    "worker startup aborted after the loopback endpoint belonged to another process",
                )
            if self.process.poll() is not None:
                self._raise_child_failure(last_error)
            try:
                response = self.client.ping()
                if self.process.poll() is not None:
                    self._raise_child_failure(last_error)
                reported_pid = response.get("pid")
                if reported_pid != self.process.pid:
                    new_pid = self.process.pid
                    self.stop(force=True)
                    raise _port_error(
                        self.database.id,
                        f"loopback port {self.database.host}:{self.database.port} answered with pid "
                        f"{reported_pid}, but the new child pid is {new_pid}",
                    )
                return
            except DiagnosticError as error:
                if self.process is None:
                    raise
                if error.code == ErrorCode.PROTO.value:
                    # A port occupied by an unrelated endpoint may answer with a
                    # malformed handshake before our child's E_PORT bind failure
                    # reaches the parent.  Give the child a short, real grace
                    # period so the authoritative bind failure wins.
                    grace_deadline = time.monotonic() + min(0.5, self.startup_timeout_seconds)
                    while time.monotonic() < grace_deadline:
                        if self.process is None:
                            raise
                        if self.process.poll() is not None:
                            self._raise_child_failure(error)
                        time.sleep(0.01)
                    self.stop(force=True)
                    raise
                if error.code == ErrorCode.SPEC.value:
                    self.stop(force=True)
                    raise
                last_error = error
                if self.process is not None and self.process.poll() is not None:
                    self._raise_child_failure(error)
                time.sleep(0.01)
            except Exception as error:  # pragma: no cover - defensive process boundary
                last_error = error
                if self.process is not None and self.process.poll() is not None:
                    self._raise_child_failure(error)
                time.sleep(0.01)
        if self.process.poll() is not None:
            self._raise_child_failure(last_error)
        self.stop(force=True)
        message = f"worker {self.database.id!r} did not complete a handshake within {self.startup_timeout_seconds}s"
        if last_error is not None:
            message += f" (last socket error: {last_error})"
        raise _timeout_error(self.database.id, message)

    def stop(self, *, force: bool = False) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            if force:
                process.kill()
            else:
                process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        self.process = None

    def kill(self) -> None:
        self.stop(force=True)

    def restart(self) -> None:
        self.stop(force=True)
        self.start()

    def __enter__(self) -> "DatabaseRuntime":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


class DatabasePool:
    def __init__(
        self,
        spec: Spec,
        spec_path: str | Path,
        *,
        startup_timeout_seconds: float = 3.0,
        socket_timeout_seconds: float = 2.0,
    ) -> None:
        self.spec = spec
        self.spec_path = spec_path
        self.startup_timeout_seconds = startup_timeout_seconds
        self.socket_timeout_seconds = socket_timeout_seconds
        self._runtimes = {
            database.id: DatabaseRuntime(
                spec_path,
                database,
                startup_timeout_seconds=startup_timeout_seconds,
                socket_timeout_seconds=socket_timeout_seconds,
            )
            for database in spec.databases
        }

    def runtime(self, db_id: str) -> DatabaseRuntime:
        try:
            return self._runtimes[db_id]
        except KeyError as error:
            raise DiagnosticError(ErrorCode.SPEC.value, f"unknown database {db_id!r}", db_id=db_id) from error

    def client(self, db_id: str) -> WorkerClient:
        return self.runtime(db_id).client

    def start_all(self, order: list[str] | None = None) -> None:
        if order is None:
            order = sorted(self._runtimes)
        for db_id in order:
            self.runtime(db_id).start()

    def stop_all(self, *, force: bool = False) -> None:
        for db_id in sorted(self._runtimes, reverse=True):
            self.runtime(db_id).stop(force=force)

    def kill_all(self) -> None:
        for db_id in sorted(self._runtimes, reverse=True):
            self.runtime(db_id).kill()

    def restart(self, db_id: str) -> None:
        self.runtime(db_id).restart()

    def __enter__(self) -> "DatabasePool":
        self.start_all()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop_all()


def force_kill_process_group(runtime: DatabaseRuntime) -> None:
    """Explicitly force-kill a worker; no graceful signal handler can run."""

    process = runtime.process
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1.0)
    if process.stdout is not None:
        process.stdout.close()
    if process.stderr is not None:
        process.stderr.close()
    runtime.process = None
