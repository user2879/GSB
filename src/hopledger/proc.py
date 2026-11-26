from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .errors import HopLedgerError, PortInUseError
from .protocol import call_jsonl, handshake
from .specs import SpecBundle


class ProcessManager:
    def __init__(self, bundle: SpecBundle, log_dir: str | Path) -> None:
        self.bundle = bundle
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.processes: dict[str, subprocess.Popen[bytes]] = {}
        self.log_files: dict[str, Any] = {}
        self.src_dir = str(Path(__file__).resolve().parents[1])

    def _node_kind(self, node_id: str) -> str:
        if node_id == self.bundle.origin.origin_id:
            return "origin"
        self.bundle.chain.hop(node_id)
        return "hop"

    def _port(self, node_id: str) -> int:
        if node_id == self.bundle.origin.origin_id:
            return self.bundle.origin.port
        return self.bundle.chain.hop(node_id).port

    def _command(self, node_id: str) -> list[str]:
        kind = self._node_kind(node_id)
        command = [
            sys.executable,
            "-m",
            "hopledger.nodes",
            "--config-dir",
            str(self.bundle.config_dir),
            "--kind",
            kind,
        ]
        if kind == "hop":
            command.extend(["--hop-id", node_id])
        return command

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = self.src_dir + (os.pathsep + existing if existing else "")
        return env

    def _read_log_error(self, node_id: str) -> dict[str, Any] | None:
        path = self.log_dir / f"{node_id}.log"
        if not path.exists():
            return None
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines[-20:]):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("level") == "error":
                return value
        return None

    def start_node(self, node_id: str, *, startup_timeout_seconds: float = 10.0) -> dict[str, Any]:
        if node_id in self.processes and self.processes[node_id].poll() is None:
            return {"node_id": node_id, "already_running": True}

        previous_log = self.log_files.pop(node_id, None)
        if previous_log is not None:
            try:
                previous_log.close()
            except OSError:
                pass
        log_path = self.log_dir / f"{node_id}.log"
        log_handle = log_path.open("ab")
        self.log_files[node_id] = log_handle
        proc = subprocess.Popen(
            self._command(node_id),
            cwd=str(Path.cwd()),
            env=self._environment(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        self.processes[node_id] = proc
        port = self._port(node_id)
        deadline = time.monotonic() + startup_timeout_seconds
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            return_code = proc.poll()
            if return_code is not None:
                error_record = self._read_log_error(node_id) or {}
                if error_record.get("code") == "E_PORT":
                    raise PortInUseError(
                        f"node {node_id!r} cannot bind 127.0.0.1:{port}",
                        hop_id=node_id,
                        field="port",
                        port=port,
                        return_code=return_code,
                    )
                raise HopLedgerError(
                    f"node {node_id!r} exited before handshake (code {return_code})",
                    code="E_NODE_EXIT",
                    hop_id=node_id,
                    return_code=return_code,
                    log=str(log_path),
                )
            try:
                ack = handshake(
                    port,
                    timeout_seconds=self.bundle.diagnostic.handshake_timeout_seconds,
                )
                ack["pid"] = proc.pid
                return ack
            except (HopLedgerError, OSError) as exc:
                last_error = exc
                # Connection refused during startup is retried. A handshake
                # timeout is also retried until the startup deadline; direct
                # handshake tests observe E_TIMEOUT without this wrapper.
                time.sleep(0.025)

        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
        raise HopLedgerError(
            f"node {node_id!r} did not become ready within {startup_timeout_seconds}s",
            code="E_START_TIMEOUT",
            hop_id=node_id,
            port=port,
            last_error=str(last_error),
        )

    def start_all(self) -> dict[str, dict[str, Any]]:
        acks: dict[str, dict[str, Any]] = {}
        acks[self.bundle.origin.origin_id] = self.start_node(self.bundle.origin.origin_id)
        for hop in self.bundle.chain.hops:
            acks[hop.hop_id] = self.start_node(hop.hop_id)
        return acks

    def call_node(self, node_id: str, message: dict[str, Any]) -> dict[str, Any]:
        return call_jsonl(
            port=self._port(node_id),
            message=message,
            timeout_seconds=self.bundle.diagnostic.timeout_seconds,
        )

    def kill_node(self, node_id: str) -> int:
        proc = self.processes.get(node_id)
        if proc is None:
            raise HopLedgerError(f"node {node_id!r} is not managed", code="E_UNKNOWN_NODE")
        if proc.poll() is None:
            proc.kill()
            return proc.wait(timeout=5)
        return proc.returncode

    def terminate_node(self, node_id: str) -> int:
        proc = self.processes[node_id]
        if proc.poll() is None:
            proc.terminate()
            try:
                return proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        return proc.wait(timeout=5)

    def restart_node(self, node_id: str) -> dict[str, Any]:
        proc = self.processes.get(node_id)
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        return self.start_node(node_id)

    def stop_all(self) -> None:
        for node_id in list(self.processes):
            try:
                self.terminate_node(node_id)
            except Exception:
                proc = self.processes[node_id]
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
        for handle in self.log_files.values():
            try:
                handle.close()
            except Exception:
                pass
        self.log_files.clear()

    def __enter__(self) -> "ProcessManager":
        self.start_all()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_all()
