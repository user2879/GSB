from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import CloudShardError, PortInUseError, ProtocolTimeoutError
from .spec import NodeSpec, Spec


SRC_DIR = Path(__file__).resolve().parents[1]


@dataclass
class NodeProcess:
    node: NodeSpec
    process: subprocess.Popen[str]
    log_handle: Any
    ready: dict[str, Any]


class NodeCluster:
    def __init__(self, spec: Spec, spec_path: str | Path, workdir: str | Path, *, ready_timeout_s: float = 15.0) -> None:
        self.spec = spec
        self.spec_path = Path(spec_path).resolve()
        self.workdir = Path(workdir)
        self.ready_timeout_s = ready_timeout_s
        self.processes: dict[str, NodeProcess] = {}

    def start_all(self) -> dict[str, NodeProcess]:
        (self.workdir / "logs").mkdir(parents=True, exist_ok=True)
        for node in self.spec.nodes:
            self.start_node(node)
        return self.processes

    def start_node(self, node: NodeSpec) -> NodeProcess:
        log_path = self.workdir / "logs" / f"{node.node_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("ab")
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        cmd = [
            sys.executable,
            "-m",
            "cloudshard.node_server",
            "--spec",
            str(self.spec_path),
            "--node-id",
            node.node_id,
        ]
        started = time.monotonic()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=log_handle,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=str(self.workdir),
        )
        assert process.stdout is not None
        remaining = max(0.1, self.ready_timeout_s - (time.monotonic() - started))
        readable, _, _ = select.select([process.stdout], [], [], remaining)
        if not readable:
            process.kill()
            process.wait(timeout=3)
            log_handle.close()
            raise ProtocolTimeoutError(
                f"node {node.node_id} did not emit ready line within {remaining:.1f}s",
                field="ready",
            )
        line = process.stdout.readline()
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            payload = {"type": "unparsed", "raw_line": line}
        if payload.get("type") == "startup_error":
            process.wait(timeout=3)
            log_handle.close()
            if payload.get("code") == "E_PORT":
                raise PortInUseError(
                    payload.get("error", f"port {node.port} is unavailable"),
                    path=payload.get("path"),
                    field=payload.get("field", "port"),
                )
            raise CloudShardError(payload.get("error", "node startup failed"))
        if payload.get("type") != "ready":
            process.kill()
            process.wait(timeout=3)
            log_handle.close()
            raise CloudShardError(f"node {node.node_id} returned unexpected startup payload: {payload}")
        record = NodeProcess(node=node, process=process, log_handle=log_handle, ready=payload)
        self.processes[node.node_id] = record
        return record

    def stop_all(self) -> None:
        for record in list(self.processes.values()):
            if record.process.poll() is None:
                record.process.terminate()
        deadline = time.monotonic() + 5
        for record in list(self.processes.values()):
            timeout = max(0.1, deadline - time.monotonic())
            try:
                record.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record.process.kill()
                record.process.wait(timeout=3)
            record.log_handle.close()
        self.processes.clear()

    def __enter__(self) -> "NodeCluster":
        return self.start_all()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_all()
