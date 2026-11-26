from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .errors import Codes, DiagnosticError
from .spec import SpecBundle


REPO_ROOT = Path(__file__).resolve().parent.parent


class NodePool:
    def __init__(self, bundle: SpecBundle, log_dir: str | Path | None = None, timeout: float = 8.0) -> None:
        self.bundle = bundle
        self.timeout = timeout
        self.log_dir = Path(log_dir or bundle.root / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_handles: dict[str, object] = {}

    def start(self) -> None:
        for node in self.bundle.nodes:
            log_path = self.log_dir / f"node-{node.id}.log"
            handle = log_path.open("wb")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "edge_shard_diag.node_worker",
                    "--node-id",
                    node.id,
                    "--host",
                    node.host,
                    "--port",
                    str(node.port),
                    "--fixed-scale",
                    str(node.fixed_scale),
                ],
                cwd=str(REPO_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            self.processes[node.id] = process
            self.log_handles[node.id] = handle
            self._wait_for_port(node.id, node.host, node.port, process, log_path)

    def _wait_for_port(self, node_id: str, host: str, port: int, process: subprocess.Popen[bytes], log_path: Path) -> None:
        deadline = time.monotonic() + self.timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1000:]
                code = Codes.PORT if f"[{Codes.PORT}]" in log_tail or "Address already in use" in log_tail else Codes.STATE
                raise DiagnosticError(code, f"node subprocess {node_id} exited code={process.returncode}; log={log_tail}", source=str(log_path))
            try:
                with socket.create_connection((host, port), timeout=0.25):
                    return
            except OSError as exc:
                last_error = exc
                time.sleep(0.05)
        raise DiagnosticError(Codes.TIMEOUT, f"timed out waiting for node {node_id} on {host}:{port}: {last_error}", field="port")

    def stop(self) -> None:
        for node_id, process in list(self.processes.items()):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        for handle in self.log_handles.values():
            try:
                handle.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        self.processes.clear()
        self.log_handles.clear()

    def __enter__(self) -> "NodePool":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()
