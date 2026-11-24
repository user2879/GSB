"""Probe subprocess lifecycle management."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .declarations import DeclarationSet
from .errors import LedgerError
from .logging import StructuredLogger


@dataclass
class ProbeProcess:
    session_id: str
    port: int
    process: subprocess.Popen[bytes]
    ready_path: Path
    stderr_path: Path


class ProbeSupervisor:
    """Starts one deterministic probe server process per declared session."""

    def __init__(
        self,
        declarations: DeclarationSet,
        *,
        work_dir: str | Path | None = None,
        logger: StructuredLogger | None = None,
        startup_timeout_s: float = 5.0,
    ) -> None:
        self.declarations = declarations
        self.logger = logger or StructuredLogger()
        self.startup_timeout_s = startup_timeout_s
        self._created_root = work_dir is None
        self.root = Path(
            tempfile.mkdtemp(prefix="probelease-probes-")
            if work_dir is None
            else work_dir
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, ProbeProcess] = {}

    @property
    def running(self) -> bool:
        return bool(self.processes)

    def start(self) -> "ProbeSupervisor":
        if self.processes:
            return self
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            for session in self.declarations.sessions:
                spec = {
                    "port": session.port,
                    "session_id": session.session_id,
                    "probes": [
                        self.declarations.probes[probe_id].canonical()
                        for probe_id in session.probe_ids
                    ],
                }
                spec_path = self.root / f"{session.session_id}.spec.json"
                ready_path = self.root / f"{session.session_id}.ready.json"
                stderr_path = self.root / f"{session.session_id}.stderr.log"
                try:
                    ready_path.unlink()
                except FileNotFoundError:
                    pass
                spec_path.write_text(
                    json.dumps(spec, sort_keys=True, separators=(",", ":")),
                    encoding="utf-8",
                )
                with stderr_path.open("wb") as stderr_handle:
                    creationflags = (
                        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                    )
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "probelease.probe_runner",
                            "--spec",
                            str(spec_path),
                            "--ready-file",
                            str(ready_path),
                            "--parent-pid",
                            str(os.getpid()),
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=stderr_handle,
                        creationflags=creationflags,
                        close_fds=True,
                    )
                self.processes[session.session_id] = ProbeProcess(
                    session_id=session.session_id,
                    port=session.port,
                    process=process,
                    ready_path=ready_path,
                    stderr_path=stderr_path,
                )
            deadline = time.monotonic() + self.startup_timeout_s
            for item in self.processes.values():
                while not item.ready_path.exists():
                    if item.process.poll() is not None:
                        stderr = _safe_read(item.stderr_path)
                        raise LedgerError(
                            "probe process exited before becoming ready",
                            "probe_process",
                            {
                                "session_id": item.session_id,
                                "port": item.port,
                                "returncode": item.process.returncode,
                                "stderr": stderr,
                            },
                            code="E_PROBE",
                        )
                    if time.monotonic() >= deadline:
                        raise LedgerError(
                            "timed out waiting for probe process readiness",
                            "probe_process",
                            {"session_id": item.session_id, "port": item.port},
                            code="E_PROBE",
                        )
                    time.sleep(0.01)
                ready = json.loads(item.ready_path.read_text(encoding="utf-8"))
                if ready.get("pid") != item.process.pid or ready.get("port") != item.port:
                    raise LedgerError(
                        "probe readiness record mismatch",
                        "probe_process",
                        {"session_id": item.session_id, "ready": ready},
                        code="E_PROBE",
                    )
                self.logger.emit(
                    "probe.ready",
                    session_id=item.session_id,
                    port=item.port,
                    probe_pid=item.process.pid,
                )
            return self
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        for item in list(self.processes.values()):
            if item.process.poll() is None:
                try:
                    item.process.terminate()
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 2.0
        for item in list(self.processes.values()):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                item.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    item.process.kill()
                except ProcessLookupError:
                    pass
                item.process.wait(timeout=1.0)
            self.logger.emit(
                "probe.stopped",
                session_id=item.session_id,
                probe_pid=item.process.pid,
                returncode=item.process.returncode,
            )
        self.processes.clear()

    def close(self) -> None:
        self.stop()
        if self._created_root:
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "ProbeSupervisor":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        return ""
