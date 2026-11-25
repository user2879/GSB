from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from .errors import E_PORT, E_TIMEOUT, MeshError
from .protocol import request_json_line
from .timeutil import monotonic_ms


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def allocate_ports(spec: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    runtime = {
        key: value
        for key, value in spec.items()
        if not key.startswith("_") and key != "declaration_digest"
    }
    used: set[int] = set()
    for node in runtime["nodes"]:
        while True:
            port = free_port()
            if port not in used:
                used.add(port)
                node["port"] = port
                break
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return runtime


def port_is_open(host: str, port: int, timeout_ms: int = 200) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_ms / 1000.0):
            return True
    except OSError:
        return False


def ensure_ports_free(spec: dict[str, Any]) -> None:
    for node in spec["nodes"]:
        if port_is_open(node["host"], int(node["port"])):
            raise MeshError(
                f"loopback port {node['port']} is already in use",
                code=E_PORT,
                field="port",
                details={"node_id": node["node_id"], "port": node["port"]},
            )


def _parent_death_signal() -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass


class NodeCluster:
    def __init__(
        self,
        spec: dict[str, Any],
        workdir: str | Path = "work",
        *,
        startup_timeout_ms: int = 10_000,
    ) -> None:
        self.spec = spec
        self.workdir = Path(workdir)
        self.run_dir = self.workdir / "nodes"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout_ms = startup_timeout_ms
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.ready_paths: dict[str, Path] = {}
        self.log_handles: dict[str, Any] = {}
        self._lock = threading.RLock()

    def start(self) -> "NodeCluster":
        ensure_ports_free(self.spec)
        for node in self.spec["nodes"]:
            self._start_one(node)
        return self

    def _start_one(self, node: dict[str, Any]) -> None:
        node_id = node["node_id"]
        ready_path = self.run_dir / f"{node_id}.ready.json"
        if ready_path.exists():
            ready_path.unlink()
        log_path = self.run_dir / f"{node_id}.log"
        log_handle = log_path.open("ab")
        env = dict(**__import__("os").environ)
        env["PYTHONPATH"] = str(PROJECT_ROOT) + (__import__("os").pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        runtime_spec = self.workdir / "runtime-declarations.json"
        runtime_spec.parent.mkdir(parents=True, exist_ok=True)
        clean = {key: value for key, value in self.spec.items() if not key.startswith("_")}
        runtime_spec.write_text(json.dumps(clean), encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "leasemesh.node_server",
            "--declarations",
            str(runtime_spec),
            "--node-id",
            node_id,
            "--ready-file",
            str(ready_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=_parent_death_signal if sys.platform.startswith("linux") else None,
        )
        self.processes[node_id] = process
        self.ready_paths[node_id] = ready_path
        self.log_handles[node_id] = log_handle

        deadline = monotonic_ms() + self.startup_timeout_ms
        while monotonic_ms() < deadline:
            if ready_path.exists():
                try:
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    ready = None
                if ready is not None:
                    if not ready.get("ok"):
                        self.stop()
                        raise MeshError(
                            ready.get("error") or "node failed to start",
                            code=ready.get("code", E_PORT),
                            field=ready.get("field"),
                            details=ready,
                        )
                    return
            exit_code = process.poll()
            if exit_code is not None:
                log_tail = self._log_tail(node_id)
                self.stop()
                raise MeshError(
                    f"node {node_id} exited with {exit_code}: {log_tail}",
                    code=E_PORT,
                    field="port",
                    details={"node_id": node_id, "log_tail": log_tail},
                )
            __import__("time").sleep(0.02)
        self.stop()
        raise MeshError(
            f"timed out waiting for node {node_id} ready handshake",
            code=E_TIMEOUT,
            field="handshake",
            details={"node_id": node_id},
        )

    def _log_tail(self, node_id: str, limit: int = 2000) -> str:
        path = self.run_dir / f"{node_id}.log"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[-limit:]
        except OSError:
            return ""

    def request(self, node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        node = next(item for item in self.spec["nodes"] if item["node_id"] == node_id)
        return request_json_line(
            node["host"],
            int(node["port"]),
            payload,
            node_id=node_id,
            timeout_ms=int(self.spec.get("socket_timeout_ms", 2000)),
            handshake_timeout_ms=int(self.spec.get("handshake_timeout_ms", 1000)),
            buffer_bytes=int(self.spec.get("io_buffer_bytes", 1_048_576)),
        )

    def stop(self) -> None:
        with self._lock:
            for node_id, process in list(self.processes.items()):
                if process.poll() is None:
                    try:
                        self.request(node_id, {"op": "shutdown", "request_id": f"shutdown-{node_id}"})
                    except MeshError:
                        try:
                            process.terminate()
                        except OSError:
                            pass
                    try:
                        process.wait(timeout=1.5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        try:
                            process.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            pass
            for handle in self.log_handles.values():
                try:
                    handle.close()
                except OSError:
                    pass
            self.processes.clear()
            self.log_handles.clear()

    def __enter__(self) -> "NodeCluster":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def list_residual_python_nodes(spec: dict[str, Any]) -> list[dict[str, Any]]:
    residual: list[dict[str, Any]] = []
    for node in spec["nodes"]:
        if port_is_open(node["host"], int(node["port"])):
            residual.append({"node_id": node["node_id"], "port": node["port"]})
    return residual
