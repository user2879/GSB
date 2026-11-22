"""Orchestration of one independent arbiter subprocess per domain.

Each domain binds its own high loopback port on 127.0.0.1 and runs an
independent Python process. The orchestrator performs real port-availability
checks, bounded handshake timeouts and clean restart semantics: a restarted
arbiter starts with an empty hold table (no inherited ownership), so recovery
must rehydrate from the committed ledger via event replay.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .errors import HandshakeError, PortError
from .models import Domain
from .protocol import (
    HANDSHAKE_TIMEOUT_MS,
    ArbiterClient,
    check_port_free,
)

ENTRY = "distlock.arbiter.process_entry"


class ArbiterProcess:
    def __init__(self, domain: Domain, host: str, port: int, scale_factor: int,
                 domain_set_version: int, runtime_dir: Path,
                 python_exe: str | None = None):
        self.domain = domain
        self.host = host
        self.port = port
        self.scale_factor = scale_factor
        self.domain_set_version = domain_set_version
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.python_exe = python_exe or sys.executable
        self.proc: subprocess.Popen | None = None
        self.client: ArbiterClient | None = None
        self.generation = 0
        self._ready_path = self.runtime_dir / f"ready_{domain.domain_id}.json"

    def start(self) -> ArbiterClient:
        self.generation += 1
        check_port_free(self.host, self.port)  # E_PORT on collision
        if self._ready_path.exists():
            self._ready_path.unlink()

        config = {
            "domain_id": self.domain.domain_id,
            "capacity": self.domain.capacity,
            "lease_duration_ms": self.domain.lease_duration_ms,
            "renew_interval_ms": self.domain.renew_interval_ms,
            "queue_limit": self.domain.queue_limit,
            "version": self.domain.version,
            "scale_factor": self.scale_factor,
            "domain_set_version": self.domain_set_version,
            "host": self.host,
            "port": self.port,
            "ready_path": str(self._ready_path),
        }
        # Windows: CREATE_NEW_PROCESS_GROUP gives an independent process group.
        kwargs: dict = {}
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(
            [self.python_exe, "-m", ENTRY, json.dumps(config, separators=(",", ":"))],
            cwd=Path(__file__).resolve().parent.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
        self._await_ready()
        try:
            self.client = ArbiterClient(
                self.host, self.port, self.domain.domain_id,
                self.domain.version, self.domain_set_version,
                handshake_timeout_ms=HANDSHAKE_TIMEOUT_MS)
        except Exception:
            self.kill()
            raise
        return self.client

    def _await_ready(self) -> None:
        assert self.proc is not None
        deadline = time.monotonic() + HANDSHAKE_TIMEOUT_MS / 1000.0
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                detail = self._read_ready()
                if detail and detail.get("status") == "E_PORT":
                    raise PortError(
                        "arbiter subprocess failed to bind port",
                        {"domain_id": self.domain.domain_id, "port": self.port,
                         "detail": detail.get("detail")})
                raise PortError(
                    "arbiter subprocess exited during startup",
                    {"domain_id": self.domain.domain_id, "port": self.port,
                     "exit_code": self.proc.returncode})
            if self._ready_path.exists():
                detail = self._read_ready()
                if detail and detail.get("status") == "READY":
                    return
                if detail and detail.get("status") == "E_PORT":
                    self.kill()
                    raise PortError("arbiter subprocess failed to bind port",
                                    {"domain_id": self.domain.domain_id,
                                     "port": self.port,
                                     "detail": detail.get("detail")})
            time.sleep(0.01)
        self.kill()
        raise HandshakeError("arbiter readiness timeout",
                             {"domain_id": self.domain.domain_id,
                              "port": self.port})

    def _read_ready(self) -> dict | None:
        try:
            return json.loads(self._ready_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def kill(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.proc = None

    def restart(self) -> ArbiterClient:
        """Force-kill then start fresh: in-memory ownership is discarded."""
        self.kill()
        return self.start()


class ArbiterPool:
    def __init__(self, domains: dict[str, Domain], host: str,
                 base_port: int, scale_factor: int, domain_set_version: int,
                 runtime_dir: str | Path | None = None,
                 python_exe: str | None = None):
        self.host = host
        self.runtime_dir = Path(runtime_dir or tempfile.mkdtemp(prefix="distlock_"))
        self.processes: dict[str, ArbiterProcess] = {}
        self.clients: dict[str, ArbiterClient] = {}
        for offset, (did, domain) in enumerate(sorted(domains.items())):
            self.processes[did] = ArbiterProcess(
                domain, host, base_port + offset, scale_factor,
                domain_set_version, self.runtime_dir, python_exe)

    def start_all(self) -> None:
        for did in sorted(self.processes):
            self.clients[did] = self.processes[did].start()

    def client(self, domain_id: str) -> ArbiterClient:
        return self.clients[domain_id]

    def restart_domain(self, domain_id: str) -> ArbiterClient:
        # Fresh generation: empty hold table, old ownership never reused.
        self.clients[domain_id] = self.processes[domain_id].restart()
        return self.clients[domain_id]

    def generations(self) -> dict[str, int]:
        return {did: p.generation for did, p in self.processes.items()}

    def shutdown(self) -> None:
        for proc in self.processes.values():
            proc.kill()
        self.clients.clear()
