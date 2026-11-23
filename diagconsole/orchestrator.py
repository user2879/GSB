"""Local orchestration of deterministic cluster simulation subprocesses."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from .errors import DiagError
from .service import REQUIRED_REPLY_FIELDS

READY_TIMEOUT_SECONDS = 8.0
REPLY_TIMEOUT_SECONDS = 8.0


class ClusterProcess:
    def __init__(self, cluster, process, ready_at):
        self.cluster = cluster
        self.process = process
        self.ready_at = ready_at


class Orchestrator:
    """Starts one service subprocess per declared cluster."""

    def __init__(self, spec):
        self.spec = spec
        self.processes: dict[str, ClusterProcess] = {}

    def start_all(self) -> None:
        for cluster in self.spec.clusters:
            self._start(cluster)

    def _start(self, cluster) -> None:
        if self._port_in_use(cluster.port):
            raise DiagError(
                "E_PORT",
                f"loopback port {cluster.port} for cluster {cluster.cluster_id!r} "
                "is already in use",
                {"cluster": cluster.cluster_id, "port": cluster.port})
        command = [
            sys.executable, "-m", "diagconsole.service",
            "--cluster-id", cluster.cluster_id,
            "--host", cluster.host,
            "--port", str(cluster.port),
        ]
        creationflags = 0
        if sys.platform == "win32":  # pragma: no cover - Windows only branch
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=creationflags,
        )
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        ready_line = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                detail = self._drain(process)
                raise DiagError(
                    "E_PORT",
                    f"service for cluster {cluster.cluster_id!r} exited during "
                    f"handshake (likely port {cluster.port} unavailable): {detail}",
                    {"cluster": cluster.cluster_id, "port": cluster.port})
            line = process.stdout.readline()
            if not line:
                time.sleep(0.02)
                continue
            try:
                ready_line = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ready_line.get("type") == "ready":
                break
            if ready_line.get("type") == "fatal":
                raise DiagError(
                    ready_line.get("code", "E_PORT"),
                    f"service for cluster {cluster.cluster_id!r} failed to bind "
                    f"port {cluster.port}",
                    {"cluster": cluster.cluster_id, "port": cluster.port})
        if ready_line is None or ready_line.get("type") != "ready":
            self._terminate(process)
            raise DiagError(
                "E_HANDSHAKE",
                f"ready handshake timed out for cluster {cluster.cluster_id!r} "
                f"on port {cluster.port}",
                {"cluster": cluster.cluster_id, "port": cluster.port})
        self.processes[cluster.cluster_id] = ClusterProcess(
            cluster, process, time.monotonic())

    def dispatch(self, request: dict) -> dict:
        cluster_id = request["cluster_id"]
        if cluster_id not in self.processes:
            raise DiagError("E_HANDSHAKE",
                            f"no live service for cluster {cluster_id!r}",
                            {"cluster": cluster_id})
        cluster = self.spec.cluster_map[cluster_id]
        payload = json.dumps(request, sort_keys=True,
                             separators=(",", ":")).encode("utf-8") + b"\n"
        try:
            with socket.create_connection((cluster.host, cluster.port),
                                          timeout=REPLY_TIMEOUT_SECONDS) as conn:
                conn.sendall(payload)
                file_obj = conn.makefile("rw", encoding="utf-8")
                raw = file_obj.readline()
        except (OSError, socket.timeout) as exc:
            raise DiagError(
                "E_HANDSHAKE",
                f"no reply from cluster {cluster_id!r}: {exc}",
                {"cluster": cluster_id}) from exc
        if not raw:
            raise DiagError("E_HANDSHAKE",
                            f"empty reply from cluster {cluster_id!r}",
                            {"cluster": cluster_id})
        try:
            reply = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DiagError("E_REPLY",
                            f"non-JSON reply from cluster {cluster_id!r}",
                            {"cluster": cluster_id}) from exc
        missing = [field for field in REQUIRED_REPLY_FIELDS if field not in reply]
        if missing:
            raise DiagError(
                "E_REPLY",
                f"reply from cluster {cluster_id!r} is missing fields {missing}",
                {"cluster": cluster_id, "missing": missing})
        if reply["cluster_id"] != cluster_id or \
                reply["task_id"] != request["task_id"] or \
                reply["version"] != request["version"]:
            raise DiagError(
                "E_REPLY",
                f"reply identity mismatch from cluster {cluster_id!r}",
                {"cluster": cluster_id,
                 "expected": {"cluster_id": cluster_id,
                              "task_id": request["task_id"],
                              "version": request["version"]},
                 "actual": {key: reply.get(key)
                            for key in ("cluster_id", "task_id", "version")}})
        return reply

    def stop_all(self) -> None:
        for entry in self.processes.values():
            self._terminate(entry.process)
        self.processes.clear()

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    @staticmethod
    def _drain(process: subprocess.Popen) -> str:
        out = process.stdout.read() if process.stdout else ""
        err = process.stderr.read() if process.stderr else ""
        return (out + err).strip()

    @staticmethod
    def _port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.3)
            return probe.connect_ex(("127.0.0.1", port)) == 0
