from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .hashing import canonical_dumps

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NodeCluster:
    def __init__(
        self,
        nodes: dict[str, dict[str, Any]],
        capacity_policies: dict[int, dict[str, Any]],
        *,
        clock_mode: str = "seeded",
        log_dir: str | Path | None = None,
        timeout: float = 8.0,
    ) -> None:
        self.nodes = nodes
        self.policies = capacity_policies
        self.clock_mode = clock_mode
        self.log_dir = Path(log_dir) if log_dir else None
        self.timeout = timeout
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_handles: dict[str, Any] = {}
        self._closed = False
        self._started = False
        atexit.register(self.close)

    def __enter__(self) -> "NodeCluster":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        policies_json = canonical_dumps({str(key): value for key, value in self.policies.items()})
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH")
        paths = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
        if existing_pythonpath:
            paths.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(paths)

        for node_id in sorted(self.nodes):
            node = self.nodes[node_id]
            if self.log_dir is not None:
                log_handle = (self.log_dir / f"{node_id}.log").open("ab")
                self.log_handles[node_id] = log_handle
                stdout = log_handle
                stderr = subprocess.STDOUT
            else:
                stdout = subprocess.DEVNULL
                stderr = subprocess.DEVNULL

            cmd = [
                sys.executable,
                "-m",
                "workbench.node_server",
                "--node-id",
                node_id,
                "--port",
                str(node["port"]),
                "--capacity-slots",
                str(node["capacity_slots"]),
                "--lease-limit",
                str(node["lease_limit"]),
                "--seed",
                str(node["seed"]),
                "--capacity-version",
                str(node["capacity_version"]),
                "--quota-version",
                str(node.get("quota_version", 1)),
                "--clock-mode",
                self.clock_mode,
                "--policies-json",
                policies_json,
                "--parent-pid",
                str(os.getpid()),
            ]
            creationflags = 0
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
            )
            self.processes[node_id] = process

        deadline = time.monotonic() + self.timeout
        for node_id in sorted(self.nodes):
            port = int(self.nodes[node_id]["port"])
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                if self.processes[node_id].poll() is not None:
                    raise RuntimeError(
                        f"node {node_id} exited early with code {self.processes[node_id].returncode}; "
                        f"see {self.log_dir / (node_id + '.log') if self.log_dir else 'stderr'}"
                    )
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                        pass
                    last_error = None
                    break
                except OSError as exc:
                    last_error = exc
                    time.sleep(0.05)
            if last_error is not None:
                raise TimeoutError(f"node {node_id} did not listen on 127.0.0.1:{port}: {last_error}")

    def request(self, node_id: str, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        node = self.nodes[node_id]
        body = dict(payload)
        body.setdefault("request_id", f"req-{os.getpid()}-{node_id}-{time.monotonic_ns()}")
        # Deterministic request ids are supplied by the engine; the fallback remains unique.
        data = (canonical_dumps(body) + "\n").encode("utf-8")
        try:
            with socket.create_connection(("127.0.0.1", int(node["port"])), timeout=timeout or self.timeout) as sock:
                sock.settimeout(timeout or self.timeout)
                sock.sendall(data)
                buffered = sock.makefile("rwb")
                line = buffered.readline()
            if not line:
                return {
                    "ok": False,
                    "request_id": body["request_id"],
                    "error": {"code": "E_PROTOCOL_VIOLATION", "message": "node closed connection without a response"},
                }
            response = json.loads(line.decode("utf-8"))
            if not isinstance(response, dict):
                raise ValueError("response is not a JSON object")
            return response
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return {
                "ok": False,
                "request_id": body["request_id"],
                "error": {"code": "E_PROTOCOL_VIOLATION", "message": f"client protocol failure: {exc}"},
            }

    def request_many(
        self,
        requests: list[tuple[str, dict[str, Any]]],
        *,
        parallel: bool = True,
    ) -> dict[str, dict[str, Any]]:
        if not parallel or len(requests) <= 1:
            return {node_id: self.request(node_id, payload) for node_id, payload in requests}
        with ThreadPoolExecutor(max_workers=min(8, len(requests))) as pool:
            futures = {
                node_id: pool.submit(self.request, node_id, payload)
                for node_id, payload in requests
            }
            return {node_id: future.result() for node_id, future in futures.items()}

    def send_raw_line(self, node_id: str, raw_line: bytes, *, timeout: float | None = None) -> bytes:
        node = self.nodes[node_id]
        with socket.create_connection(("127.0.0.1", int(node["port"])), timeout=timeout or self.timeout) as sock:
            sock.settimeout(timeout or self.timeout)
            sock.sendall(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
            return sock.makefile("rb").readline()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._started = False
        for node_id in sorted(self.processes):
            process = self.processes[node_id]
            if process.poll() is None:
                try:
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                            capture_output=True,
                            check=False,
                        )
                    else:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            process.kill()
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=1)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
        for handle in self.log_handles.values():
            try:
                handle.close()
            except Exception:
                pass

    def pids(self) -> dict[str, int]:
        return {node_id: process.pid for node_id, process in self.processes.items()}

    def alive(self, node_id: str) -> bool:
        process = self.processes.get(node_id)
        return process is not None and process.poll() is None
