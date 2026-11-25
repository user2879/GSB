from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .protocol import NodeClient, NodeProtocolError
from .spec import NodeSpec, SpecBundle


SRC_DIR = Path(__file__).resolve().parents[1]


class NodeProcess:
    def __init__(self, spec: SpecBundle, node: NodeSpec, log_dir: str | Path | None = None):
        self.spec = spec
        self.node = node
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.proc: subprocess.Popen[bytes] | None = None
        self.log_file: Any | None = None

    def start(self, ready_timeout_s: float = 10.0) -> "NodeProcess":
        if self.proc is not None and self.proc.poll() is None:
            return self
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing if existing else "")
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = (self.log_dir / f"{self.node.node_id}.log").open("ab")
            stdout = self.log_file
            stderr = subprocess.STDOUT
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "edge_ledger.node_worker",
            "--spec-dir",
            str(self.spec.root.resolve()),
            "--node-id",
            self.node.node_id,
        ]
        self.proc = subprocess.Popen(cmd, cwd=str(Path.cwd()), env=env, stdout=stdout, stderr=stderr)
        deadline = time.monotonic() + ready_timeout_s
        client = NodeClient(self.node, self.spec.policy.socket_timeout_ms)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise NodeProtocolError(
                    f"node {self.node.node_id} exited early with code {self.proc.returncode}",
                    code="E_NODE_EXITED",
                )
            try:
                client.ping()
                return self
            except NodeProtocolError as exc:
                last_error = exc
                time.sleep(0.02)
        raise NodeProtocolError(f"node {self.node.node_id} did not become ready: {last_error}", code="E_NODE_TIMEOUT")

    def kill_hard(self) -> int:
        if self.proc is None:
            return 0
        pid = self.proc.pid
        self.proc.kill()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.wait()
        self._close_log()
        return pid

    def terminate(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.kill_hard()
        self._close_log()

    def _close_log(self) -> None:
        if self.log_file is not None:
            try:
                self.log_file.close()
            except OSError:
                pass
            self.log_file = None

    def __enter__(self) -> "NodeProcess":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.terminate()


class NodeCluster:
    def __init__(self, spec: SpecBundle, log_dir: str | Path | None = None):
        self.spec = spec
        self.log_dir = log_dir
        self.processes = {
            node.node_id: NodeProcess(spec, node, log_dir)
            for node in sorted(spec.nodes, key=lambda n: n.node_id)
        }

    def start_all(self) -> "NodeCluster":
        for process in self.processes.values():
            process.start()
        return self

    def kill_hard(self, node_id: str) -> int:
        return self.processes[node_id].kill_hard()

    def restart(self, node_id: str) -> NodeProcess:
        process = self.processes[node_id]
        process.terminate()
        process.start()
        return process

    def stop_all(self) -> None:
        for process in self.processes.values():
            process.terminate()

    def __enter__(self) -> "NodeCluster":
        return self.start_all()

    def __exit__(self, *exc: object) -> None:
        self.stop_all()
