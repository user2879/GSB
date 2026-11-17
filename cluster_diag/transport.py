from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .errors import DiagError, E_PORT, E_SERVICE, StructuredLogger
from .spec import SystemSpec
from .util import canonical_json, read_json


def is_port_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name != "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        elif hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


class ServiceClient:
    def __init__(self, host: str, port: int, cluster_id: str, session: str, *, timeout: float = 10.0) -> None:
        self.host = host
        self.port = port
        self.cluster_id = cluster_id
        self.session = session
        self.timeout = timeout

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        wire = (canonical_json(payload) + "\n").encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
                sock.settimeout(self.timeout)
                sock.sendall(wire)
                reader = sock.makefile("rb")
                response_line = reader.readline()
        except OSError as exc:
            raise DiagError(
                E_PORT,
                f"simulator connection failed for {self.cluster_id} at {self.host}:{self.port}: {exc}",
                field_path=f"$.clusters[cluster_id={self.cluster_id}].port",
                details={"cluster_id": self.cluster_id, "host": self.host, "port": self.port},
            ) from exc
        if not response_line:
            raise DiagError(
                E_SERVICE,
                f"simulator for {self.cluster_id} closed the connection without a response",
                details={"cluster_id": self.cluster_id},
            )
        try:
            response = json.loads(response_line.decode("utf-8"))
        except Exception as exc:
            raise DiagError(
                E_SERVICE,
                f"simulator for {self.cluster_id} returned invalid JSON: {exc}",
                details={"cluster_id": self.cluster_id},
            ) from exc
        if response.get("type") == "error":
            raise DiagError(
                response.get("code", E_SERVICE),
                response.get("message", "simulator request failed"),
                field_path=response.get("field_path"),
                details={"cluster_id": self.cluster_id, **response.get("details", {})},
            )
        return response

    def ping(self) -> dict[str, Any]:
        response = self.request({"command": "ping", "request_id": "startup-ping"})
        if (
            response.get("type") != "pong"
            or response.get("session") != self.session
            or response.get("cluster_id") != self.cluster_id
        ):
            raise DiagError(
                E_PORT,
                f"port {self.port} is occupied by a different service",
                field_path=f"$.clusters[cluster_id={self.cluster_id}].port",
                details={
                    "cluster_id": self.cluster_id,
                    "expected_session": self.session,
                    "observed": response,
                },
            )
        return response

    def distribute(self, revision: int, task_id: str, request_id: str) -> dict[str, Any]:
        response = self.request(
            {
                "command": "distribute",
                "request_id": request_id,
                "revision": revision,
                "task_id": task_id,
            }
        )
        if response.get("type") != "result":
            raise DiagError(
                E_SERVICE,
                "simulator returned an unexpected response type",
                details={"cluster_id": self.cluster_id, "response_type": response.get("type")},
            )
        if (
            response.get("cluster_id") != self.cluster_id
            or response.get("revision") != revision
            or response.get("task_id") != task_id
            or response.get("session") != self.session
        ):
            raise DiagError(
                E_SERVICE,
                "simulator response identity mismatch",
                details={"cluster_id": self.cluster_id, "revision": revision, "response": response},
            )
        return response


class ServiceManager:
    """Owns one local simulator child process per declared cluster."""

    def __init__(
        self,
        spec: SystemSpec,
        runtime_dir: str | Path,
        logger: StructuredLogger,
        *,
        inject_port_conflict: bool = False,
    ) -> None:
        self.spec = spec
        self.runtime_dir = Path(runtime_dir)
        self.logger = logger
        self.session = uuid.uuid4().hex
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.clients: dict[str, ServiceClient] = {}
        self.inject_port_conflict = inject_port_conflict
        self._blocker: socket.socket | None = None

    def _hold_conflict_port(self) -> None:
        cluster = self.spec.clusters[0]
        family = socket.AF_INET6 if ":" in cluster.host else socket.AF_INET
        blocker = socket.socket(family, socket.SOCK_STREAM)
        if os.name != "nt":
            blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        elif hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            blocker.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        if family == socket.AF_INET6:
            blocker.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        blocker.bind((cluster.host, cluster.port))
        blocker.listen(1)
        self._blocker = blocker
        self.logger.info(
            "INJECT_PORT_CONFLICT",
            "holding a declared simulator port to inject E_PORT",
            details={"cluster_id": cluster.id, "host": cluster.host, "port": cluster.port},
        )

    def _service_log_path(self, cluster_id: str) -> Path:
        safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in cluster_id)
        return self.runtime_dir / "logs" / f"service-{safe}.jsonl"

    def start(self) -> None:
        if self.inject_port_conflict:
            self._hold_conflict_port()

        for cluster in self.spec.clusters:
            if not is_port_available(cluster.host, cluster.port):
                self.close()
                raise DiagError(
                    E_PORT,
                    f"declared simulator port is already occupied: {cluster.host}:{cluster.port}",
                    field_path=f"$.clusters[cluster_id={cluster.id}].port",
                    details={"cluster_id": cluster.id, "host": cluster.host, "port": cluster.port},
                )

        project_root = str(Path(__file__).resolve().parent.parent)
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        for cluster in self.spec.clusters:
            service_log = self._service_log_path(cluster.id)
            service_log.parent.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "cluster_diag.service",
                    "--spec",
                    str(self.spec.source_path),
                    "--cluster",
                    cluster.id,
                    "--session",
                    self.session,
                    "--log",
                    str(service_log),
                ],
                cwd=project_root,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self.processes[cluster.id] = process
            client = ServiceClient(cluster.host, cluster.port, cluster.id, self.session)
            self.clients[cluster.id] = client

        deadline = time.monotonic() + 8.0
        pending = set(self.clients)
        while pending and time.monotonic() < deadline:
            for cluster_id in sorted(pending):
                process = self.processes[cluster_id]
                if process.poll() is not None:
                    stderr = b""
                    if process.stderr is not None:
                        stderr = process.stderr.read() or b""
                    details: dict[str, Any] = {
                        "cluster_id": cluster_id,
                        "exit_code": process.returncode,
                        "service_log": str(self._service_log_path(cluster_id)),
                    }
                    if stderr:
                        details["stderr"] = stderr.decode("utf-8", errors="replace")[-2000:]
                    self.close()
                    raise DiagError(
                        E_PORT if process.returncode == 3 else E_SERVICE,
                        f"simulator child failed to start for cluster {cluster_id!r}",
                        field_path=f"$.clusters[cluster_id={cluster_id}].port",
                        details=details,
                    )
                try:
                    self.clients[cluster_id].ping()
                except DiagError:
                    continue
                pending.remove(cluster_id)
            if pending:
                time.sleep(0.05)
        if pending:
            self.close()
            raise DiagError(
                E_SERVICE,
                "simulator startup timed out",
                details={"clusters": sorted(pending), "timeout_seconds": 8},
            )
        self.logger.info(
            "SERVICES_READY",
            "all deterministic simulators are ready",
            details={"clusters": sorted(self.clients), "session": self.session},
        )

    def client(self, cluster_id: str) -> ServiceClient:
        try:
            return self.clients[cluster_id]
        except KeyError as exc:
            raise DiagError(
                E_SERVICE,
                f"no simulator is running for cluster {cluster_id!r}",
                details={"cluster_id": cluster_id},
            ) from exc

    def close(self) -> None:
        for cluster_id, client in list(self.clients.items()):
            try:
                client.request({"command": "shutdown", "request_id": "shutdown"})
            except Exception:
                pass
        deadline = time.monotonic() + 2.0
        for cluster_id, process in list(self.processes.items()):
            try:
                remaining = max(0.05, deadline - time.monotonic())
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
            finally:
                if process.stderr is not None:
                    try:
                        process.stderr.close()
                    except Exception:
                        pass
        self.processes.clear()
        self.clients.clear()
        if self._blocker is not None:
            try:
                self._blocker.close()
            except Exception:
                pass
            self._blocker = None

    def __enter__(self) -> "ServiceManager":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
